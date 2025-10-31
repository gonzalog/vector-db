"""Flat (brute-force) vector index implementation."""

import heapq
from typing import Any
from uuid import UUID

import numpy as np

from vector_db.core.locks import ReadWriteLock
from vector_db.indexes.base import VectorIndex
from vector_db.indexes.distance import DistanceMetric, compute_batch_distances, get_distance_function
from vector_db.models import Chunk


class FlatIndex(VectorIndex):
    """
    Flat index that performs brute-force search.

    This is the simplest index that computes distances to all vectors.
    It's accurate but has O(n) search complexity.
    """

    def __init__(self, distance_metric: DistanceMetric = DistanceMetric.COSINE):
        """
        Initialize flat index.

        Args:
            distance_metric: Distance metric to use
        """
        super().__init__(distance_metric)
        self._chunks: dict[UUID, Chunk] = {}
        self._lock = ReadWriteLock()

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
            self._chunks[chunk.id] = chunk

    def add_batch(self, chunks: list[Chunk]) -> None:
        """
        Add multiple chunks at once.

        Args:
            chunks: List of chunks to add

        Raises:
            ValueError: If any embedding dimension doesn't match
        """
        for chunk in chunks:
            self._validate_dimension(chunk.embedding)

        with self._lock.write():
            for chunk in chunks:
                self._chunks[chunk.id] = chunk

    def search(
        self,
        query_embedding: list[float],
        k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """
        Search for k nearest neighbors using brute-force with NumPy optimization.

        Args:
            query_embedding: Query vector
            k: Number of results to return
            metadata_filter: Optional metadata filters

        Returns:
            List of (chunk, distance) tuples sorted by distance

        Raises:
            ValueError: If query dimension doesn't match index dimension
        """
        self._validate_dimension(query_embedding)

        with self._lock.read():
            if not self._chunks:
                return []

            # Filter chunks by metadata first
            filtered_chunks = [
                chunk
                for chunk in self._chunks.values()
                if self._matches_filter(chunk, metadata_filter)
            ]

            if not filtered_chunks:
                return []

            # Compute distances using centralized batch distance function
            distances = compute_batch_distances(
                query_embedding,
                [chunk.embedding for chunk in filtered_chunks],
                self.distance_metric,
            )

            # Get k smallest indices
            if len(distances) <= k:
                # Sort all
                indices = np.argsort(distances)
            else:
                # Get k smallest using partition (faster than full sort)
                indices = np.argpartition(distances, k - 1)[:k]
                # Sort just these k elements
                indices = indices[np.argsort(distances[indices])]

            # Return chunks with their distances
            return [
                (filtered_chunks[idx], float(distances[idx])) for idx in indices
            ]

    def remove(self, chunk_id: UUID) -> bool:
        """
        Remove a chunk from the index.

        Args:
            chunk_id: ID of chunk to remove

        Returns:
            True if removed, False if not found
        """
        with self._lock.write():
            if chunk_id in self._chunks:
                del self._chunks[chunk_id]
                return True
            return False

    def clear(self) -> None:
        """Remove all chunks from the index."""
        with self._lock.write():
            self._chunks.clear()
            self._dimension = None

    def size(self) -> int:
        """
        Get number of chunks in index.

        Returns:
            Number of chunks
        """
        with self._lock.read():
            return len(self._chunks)
