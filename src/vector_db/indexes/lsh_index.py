"""LSH (Locality-Sensitive Hashing) vector index implementation."""

import heapq
from typing import Any
from uuid import UUID

import numpy as np

from vector_db.core.locks import ReadWriteLock
from vector_db.indexes.base import VectorIndex
from vector_db.indexes.distance import DistanceMetric, get_distance_function
from vector_db.models import Chunk


class Hyperplane:
    """
    Represents a single random hyperplane for LSH hashing.

    A hyperplane divides the vector space into two half-spaces.
    Used for random projection-based hashing.
    """

    def __init__(self, dimension: int, seed: int = 42):
        """
        Initialize a random hyperplane.

        Args:
            dimension: Vector dimension
            seed: Random seed for reproducibility
        """
        self.dimension = dimension

        # Generate random hyperplane normal vector
        rng = np.random.default_rng(seed=seed)
        self.normal = rng.normal(0, 1, dimension).astype(np.float32)

        # Normalize to unit vector
        self.normal = self.normal / (np.linalg.norm(self.normal) + 1e-8)

    def project(self, embedding: list[float]) -> bool:
        """
        Project a vector onto the hyperplane and return which side it's on.

        Args:
            embedding: Vector to project

        Returns:
            True if on positive side, False if on negative side
        """
        vec = np.asarray(embedding, dtype=np.float32)

        # Compute dot product with hyperplane normal
        projection = np.dot(self.normal, vec)

        # Return which side of the hyperplane the vector is on
        return projection > 0


class HashTable:
    """
    Hash table for LSH with its own set of hyperplanes.

    Combines hashing (via hyperplanes) and bucket storage.
    Each hash table maintains independent hyperplanes for LSH.
    """

    def __init__(self, n_bits: int, dimension: int, seed: int):
        """
        Initialize hash table with hyperplanes.

        Args:
            n_bits: Number of hyperplanes (hash size in bits)
            dimension: Vector dimension
            seed: Random seed for hyperplane generation
        """
        self._buckets: dict[int, list[Chunk]] = {}
        self._hyperplanes: list[Hyperplane] = []

        # Create n_bits independent hyperplanes
        for bit_id in range(n_bits):
            hyperplane = Hyperplane(dimension=dimension, seed=seed + bit_id)
            self._hyperplanes.append(hyperplane)

    def hash(self, embedding: list[float]) -> int:
        """
        Compute hash value for an embedding.

        Args:
            embedding: Vector to hash

        Returns:
            Integer hash value
        """
        # Project onto each hyperplane to get binary bits
        bits = [hyperplane.project(embedding) for hyperplane in self._hyperplanes]

        # Convert bit vector to integer
        hash_value = 0
        for i, bit in enumerate(bits):
            if bit:
                hash_value |= (1 << i)

        return hash_value

    def add(self, hash_value: int, chunk: Chunk) -> None:
        """
        Add a chunk to a bucket.

        Args:
            hash_value: Hash value (bucket identifier)
            chunk: Chunk to add
        """
        if hash_value not in self._buckets:
            self._buckets[hash_value] = []
        self._buckets[hash_value].append(chunk)

    def get_bucket(self, hash_value: int) -> list[Chunk]:
        """
        Get all chunks in a bucket.

        Args:
            hash_value: Hash value (bucket identifier)

        Returns:
            List of chunks in the bucket (empty if bucket doesn't exist)
        """
        return self._buckets.get(hash_value, [])

    def remove_from_bucket(self, hash_value: int, chunk_id: UUID) -> bool:
        """
        Remove a chunk from a bucket.

        Args:
            hash_value: Hash value (bucket identifier)
            chunk_id: ID of chunk to remove

        Returns:
            True if removed, False if not found
        """
        if hash_value not in self._buckets:
            return False

        bucket = self._buckets[hash_value]
        self._buckets[hash_value] = [
            chunk for chunk in bucket if chunk.id != chunk_id
        ]

        # Clean up empty buckets
        if not self._buckets[hash_value]:
            del self._buckets[hash_value]

        return len(self._buckets.get(hash_value, [])) < len(bucket)

    def clear(self) -> None:
        """Remove all buckets."""
        self._buckets.clear()

    def __len__(self) -> int:
        """Get number of buckets."""
        return len(self._buckets)


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

        # Hash tables will be initialized when first vector is added
        # (we need to know the dimension first)
        self._hash_tables: list[HashTable] | None = None

        # Chunk ID to bucket hashes mapping for removal
        self._chunk_to_buckets: dict[UUID, list[int]] = {}

        self._lock = ReadWriteLock()
        self._distance_fn = get_distance_function(distance_metric)

    def _initialize_hash_tables(self, dimension: int) -> None:
        """
        Initialize hash tables with hyperplanes.

        Args:
            dimension: Vector dimension
        """
        self._hash_tables = []

        for table_id in range(self.n_hash_tables):
            # Create hash table with its own set of hyperplanes
            # Use different seed for each table to ensure independence
            seed = 42 + table_id * self.n_hash_bits
            hash_table = HashTable(
                n_bits=self.n_hash_bits,
                dimension=dimension,
                seed=seed
            )
            self._hash_tables.append(hash_table)

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
            # Initialize hash tables on first add
            if self._hash_tables is None:
                self._initialize_hash_tables(len(chunk.embedding))

            # Compute hash for each table and add to corresponding buckets
            bucket_hashes = []
            for hash_table in self._hash_tables:
                hash_value = hash_table.hash(chunk.embedding)
                bucket_hashes.append(hash_value)

                # Add to hash table
                hash_table.add(hash_value, chunk)

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
            # Initialize hash tables on first add
            if self._hash_tables is None:
                self._initialize_hash_tables(len(chunks[0].embedding))

            for chunk in chunks:
                # Compute hash for each table and add to corresponding buckets
                bucket_hashes = []
                for hash_table in self._hash_tables:
                    hash_value = hash_table.hash(chunk.embedding)
                    bucket_hashes.append(hash_value)

                    # Add to hash table
                    hash_table.add(hash_value, chunk)

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
            if self._hash_tables is None:
                raise ValueError("Index must have vectors before searching")

            if len(self._chunk_to_buckets) == 0:
                return []

            # Collect candidates from all hash tables
            candidates: set[UUID] = set()
            chunk_map: dict[UUID, Chunk] = {}

            for hash_table in self._hash_tables:
                hash_value = hash_table.hash(query_embedding)

                # Get chunks from this bucket
                bucket = hash_table.get_bucket(hash_value)
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
                self._hash_tables[table_id].remove_from_bucket(hash_value, chunk_id)

            del self._chunk_to_buckets[chunk_id]
            return True

    def clear(self) -> None:
        """Remove all chunks from the index."""
        with self._lock.write():
            if self._hash_tables is not None:
                for table in self._hash_tables:
                    table.clear()
            self._chunk_to_buckets.clear()
            self._hash_tables = None
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
