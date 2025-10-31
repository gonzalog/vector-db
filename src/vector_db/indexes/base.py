"""Base interface for vector indexes."""

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

from vector_db.indexes.distance import DistanceMetric
from vector_db.models import Chunk


class VectorIndex(ABC):
    """Abstract base class for vector indexes."""

    def __init__(self, distance_metric: DistanceMetric = DistanceMetric.COSINE):
        """
        Initialize the index.

        Args:
            distance_metric: Distance metric to use for similarity search
        """
        self.distance_metric = distance_metric
        self._dimension: int | None = None

    @abstractmethod
    def add(self, chunk: Chunk) -> None:
        """
        Add a chunk to the index.

        Args:
            chunk: Chunk to add with its embedding

        Raises:
            ValueError: If chunk embedding dimension doesn't match index dimension
        """
        pass

    @abstractmethod
    def add_batch(self, chunks: list[Chunk]) -> None:
        """
        Add multiple chunks to the index at once.

        Args:
            chunks: List of chunks to add

        Raises:
            ValueError: If any chunk embedding dimension doesn't match index dimension
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """
        Search for the k nearest neighbors.

        Args:
            query_embedding: Query vector
            k: Number of nearest neighbors to return
            metadata_filter: Optional metadata filters to apply

        Returns:
            List of (chunk, distance) tuples sorted by distance (ascending)

        Raises:
            ValueError: If query embedding dimension doesn't match index dimension
        """
        pass

    @abstractmethod
    def remove(self, chunk_id: UUID) -> bool:
        """
        Remove a chunk from the index.

        Args:
            chunk_id: ID of chunk to remove

        Returns:
            True if chunk was removed, False if not found
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Remove all chunks from the index."""
        pass

    @abstractmethod
    def size(self) -> int:
        """
        Get the number of chunks in the index.

        Returns:
            Number of chunks
        """
        pass

    def _validate_dimension(self, embedding: list[float]) -> None:
        """
        Validate embedding dimension matches index dimension.

        Args:
            embedding: Embedding to validate

        Raises:
            ValueError: If dimension doesn't match
        """
        if self._dimension is None:
            self._dimension = len(embedding)
        elif len(embedding) != self._dimension:
            raise ValueError(
                f"Embedding dimension {len(embedding)} doesn't match "
                f"index dimension {self._dimension}"
            )

    def _matches_filter(self, chunk: Chunk, metadata_filter: dict[str, Any] | None) -> bool:
        """
        Check if a chunk matches the metadata filter.

        Args:
            chunk: Chunk to check
            metadata_filter: Filter to apply (None means no filtering)

        Returns:
            True if chunk matches filter or filter is None
        """
        if metadata_filter is None:
            return True

        for key, value in metadata_filter.items():
            # Handle nested metadata access with dot notation
            if "." in key:
                parts = key.split(".")
                current = chunk.metadata.model_dump()
                try:
                    for part in parts:
                        current = current[part]
                    chunk_value = current
                except (KeyError, TypeError):
                    return False
            else:
                # Try to get value from defined metadata fields first
                chunk_dict = chunk.metadata.model_dump()

                # Check defined fields (author, tags, source, etc.)
                if key in chunk_dict and key != "custom":
                    chunk_value = chunk_dict[key]
                # Check custom fields
                elif key in chunk_dict.get("custom", {}):
                    chunk_value = chunk_dict["custom"][key]
                else:
                    # Key not found in metadata
                    return False

            # Handle list values (check if filter value is in list)
            if isinstance(chunk_value, list):
                if isinstance(value, list):
                    # All filter values must be in chunk values
                    if not all(v in chunk_value for v in value):
                        return False
                else:
                    # Single filter value must be in chunk values
                    if value not in chunk_value:
                        return False
            else:
                # Exact match
                if chunk_value != value:
                    return False

        return True
