"""LSH (Locality-Sensitive Hashing) vector index implementation."""

import heapq
from typing import Any
from uuid import UUID

import numpy as np

from vector_db.core.locks import ReadWriteLock
from vector_db.indexes.base import VectorIndex
from vector_db.indexes.distance import DistanceMetric, get_distance_function
from vector_db.models import Chunk


class LSHIndex(VectorIndex):
    """
    LSH (Locality-Sensitive Hashing) index using random hyperplanes.

    Uses random projection method to hash vectors into buckets.
    Similar vectors are likely to be hashed to the same bucket.

    This provides sublinear search time for approximate nearest neighbor search.
    """

    def __init__(
        self,
        n_hash_tables: int = 5,
        n_hash_bits: int = 8,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
    ):
        """
        Initialize LSH index.

        Args:
            n_hash_tables: Number of independent hash tables (higher = better recall)
            n_hash_bits: Number of bits per hash (higher = more buckets, lower collision)
            distance_metric: Distance metric to use
        """
        super().__init__(distance_metric)
        self.n_hash_tables = n_hash_tables
        self.n_hash_bits = n_hash_bits

        # Hash tables: table_id -> bucket_hash -> list of chunks
        self._hash_tables: list[dict[int, list[Chunk]]] = [
            {} for _ in range(n_hash_tables)
        ]

        # Random hyperplanes for each hash table
        # Will be initialized when first vector is added
        self._hyperplanes: list[np.ndarray] | None = None

        # Chunk ID to bucket hashes mapping for removal
        self._chunk_to_buckets: dict[UUID, list[int]] = {}

        self._lock = ReadWriteLock()
        self._distance_fn = get_distance_function(distance_metric)

    def _initialize_hyperplanes(self, dimension: int) -> None:
        """
        Initialize random hyperplanes for LSH.

        Args:
            dimension: Vector dimension
        """
        self._hyperplanes = []
        rng = np.random.default_rng(seed=42)  # Fixed seed for reproducibility

        for _ in range(self.n_hash_tables):
            # Create n_hash_bits random hyperplanes
            hyperplanes = rng.normal(0, 1, (self.n_hash_bits, dimension)).astype(
                np.float32
            )
            # Normalize hyperplanes
            hyperplanes = hyperplanes / (
                np.linalg.norm(hyperplanes, axis=1, keepdims=True) + 1e-8
            )
            self._hyperplanes.append(hyperplanes)

    def _compute_hash(
        self, embedding: list[float], table_id: int
    ) -> int:
        """
        Compute LSH hash for an embedding in a specific table.

        Args:
            embedding: Embedding vector
            table_id: Hash table ID

        Returns:
            Hash value (integer representation of bit vector)
        """
        vec = np.asarray(embedding, dtype=np.float32)
        hyperplanes = self._hyperplanes[table_id]

        # Compute dot products with all hyperplanes
        projections = np.dot(hyperplanes, vec)

        # Convert to binary: 1 if positive, 0 if negative
        bits = (projections > 0).astype(np.int32)

        # Convert bit vector to integer
        hash_value = int(np.dot(bits, 1 << np.arange(self.n_hash_bits)))
        return hash_value

    def add(self, chunk: Chunk) -> None:
        """
        Add a chunk to the index.

        Args:
            chunk: Chunk to add

        Raises:
            ValueError: If embedding dimension doesn't match
        """
        self._validate_dimension(chunk.embedding)

        with self._lock.write():
            # Initialize hyperplanes on first add
            if self._hyperplanes is None:
                self._initialize_hyperplanes(len(chunk.embedding))

            # Compute hash for each table and add to corresponding buckets
            bucket_hashes = []
            for table_id in range(self.n_hash_tables):
                hash_value = self._compute_hash(chunk.embedding, table_id)
                bucket_hashes.append(hash_value)

                # Add to bucket
                if hash_value not in self._hash_tables[table_id]:
                    self._hash_tables[table_id][hash_value] = []
                self._hash_tables[table_id][hash_value].append(chunk)

            # Store mapping for removal
            self._chunk_to_buckets[chunk.id] = bucket_hashes

    def add_batch(self, chunks: list[Chunk]) -> None:
        """
        Add multiple chunks at once.

        Args:
            chunks: List of chunks to add

        Raises:
            ValueError: If any embedding dimension doesn't match
        """
        if not chunks:
            return

        for chunk in chunks:
            self._validate_dimension(chunk.embedding)

        with self._lock.write():
            # Initialize hyperplanes on first add
            if self._hyperplanes is None:
                self._initialize_hyperplanes(len(chunks[0].embedding))

            for chunk in chunks:
                # Compute hash for each table and add to corresponding buckets
                bucket_hashes = []
                for table_id in range(self.n_hash_tables):
                    hash_value = self._compute_hash(chunk.embedding, table_id)
                    bucket_hashes.append(hash_value)

                    # Add to bucket
                    if hash_value not in self._hash_tables[table_id]:
                        self._hash_tables[table_id][hash_value] = []
                    self._hash_tables[table_id][hash_value].append(chunk)

                # Store mapping for removal
                self._chunk_to_buckets[chunk.id] = bucket_hashes

    def search(
        self,
        query_embedding: list[float],
        k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """
        Search for k nearest neighbors.

        Args:
            query_embedding: Query vector
            k: Number of results to return
            metadata_filter: Optional metadata filters

        Returns:
            List of (chunk, distance) tuples sorted by distance

        Raises:
            ValueError: If query dimension doesn't match or index not initialized
        """
        self._validate_dimension(query_embedding)

        with self._lock.read():
            if self._hyperplanes is None:
                raise ValueError("Index must have vectors before searching")

            if len(self._chunk_to_buckets) == 0:
                return []

            # Collect candidates from all hash tables
            candidates: set[UUID] = set()
            chunk_map: dict[UUID, Chunk] = {}

            for table_id in range(self.n_hash_tables):
                hash_value = self._compute_hash(query_embedding, table_id)

                # Get chunks from this bucket
                bucket = self._hash_tables[table_id].get(hash_value, [])
                for chunk in bucket:
                    if chunk.id not in candidates:
                        candidates.add(chunk.id)
                        chunk_map[chunk.id] = chunk

            if not candidates:
                return []

            # Compute distances and apply filters
            distances: list[tuple[float, Chunk]] = []
            for chunk_id in candidates:
                chunk = chunk_map[chunk_id]
                if not self._matches_filter(chunk, metadata_filter):
                    continue

                distance = self._distance_fn(query_embedding, chunk.embedding)
                distances.append((distance, chunk))

            # Return k smallest
            if len(distances) <= k:
                distances.sort(key=lambda x: x[0])
                return [(chunk, dist) for dist, chunk in distances]

            k_smallest = heapq.nsmallest(k, distances, key=lambda x: x[0])
            return [(chunk, dist) for dist, chunk in k_smallest]

    def remove(self, chunk_id: UUID) -> bool:
        """
        Remove a chunk from the index.

        Args:
            chunk_id: ID of chunk to remove

        Returns:
            True if removed, False if not found
        """
        with self._lock.write():
            if chunk_id not in self._chunk_to_buckets:
                return False

            bucket_hashes = self._chunk_to_buckets[chunk_id]

            # Remove from all buckets
            for table_id, hash_value in enumerate(bucket_hashes):
                bucket = self._hash_tables[table_id].get(hash_value, [])
                self._hash_tables[table_id][hash_value] = [
                    chunk for chunk in bucket if chunk.id != chunk_id
                ]

                # Clean up empty buckets
                if not self._hash_tables[table_id][hash_value]:
                    del self._hash_tables[table_id][hash_value]

            del self._chunk_to_buckets[chunk_id]
            return True

    def clear(self) -> None:
        """Remove all chunks from the index."""
        with self._lock.write():
            self._hash_tables = [{} for _ in range(self.n_hash_tables)]
            self._chunk_to_buckets.clear()
            self._hyperplanes = None
            self._dimension = None

    def size(self) -> int:
        """
        Get number of chunks in index.

        Returns:
            Number of chunks
        """
        with self._lock.read():
            return len(self._chunk_to_buckets)

    def __getstate__(self) -> dict:
        """
        Get state for pickling.

        Excludes the lock since it can't be pickled.
        """
        state = self.__dict__.copy()
        # Remove the lock - it will be recreated when unpickling
        state.pop('_lock', None)
        return state

    def __setstate__(self, state: dict) -> None:
        """
        Set state for unpickling.

        Recreates the lock.
        """
        self.__dict__.update(state)
        # Recreate the lock
        self._lock = ReadWriteLock()
