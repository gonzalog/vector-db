"""In-memory repository for storing vector database data."""

from datetime import datetime
from threading import RLock
from typing import Generic, TypeVar
from uuid import UUID

from vector_db.core.exceptions import AlreadyExistsException, NotFoundException
from vector_db.models import Chunk, Document, Library

T = TypeVar("T", Library, Document, Chunk)


class InMemoryRepository(Generic[T]):
    """Thread-safe in-memory repository using RLock for concurrency control."""

    def __init__(self):
        """Initialize the repository with a dictionary and RLock."""
        self._data: dict[UUID, T] = {}
        self._lock = RLock()

    def create(self, entity: T) -> T:
        """
        Create a new entity.

        Args:
            entity: The entity to create

        Returns:
            The created entity

        Raises:
            AlreadyExistsException: If entity with same ID already exists
        """
        with self._lock:
            if entity.id in self._data:
                raise AlreadyExistsException(
                    f"Entity with id {entity.id} already exists",
                    {"id": str(entity.id)},
                )
            self._data[entity.id] = entity
            return entity

    def get(self, entity_id: UUID) -> T:
        """
        Get an entity by ID.

        Args:
            entity_id: The ID of the entity to retrieve

        Returns:
            The entity

        Raises:
            NotFoundException: If entity not found
        """
        with self._lock:
            if entity_id not in self._data:
                raise NotFoundException(
                    f"Entity with id {entity_id} not found",
                    {"id": str(entity_id)},
                )
            return self._data[entity_id]

    def get_all(self) -> list[T]:
        """
        Get all entities.

        Returns:
            List of all entities
        """
        with self._lock:
            return list(self._data.values())

    def update(self, entity_id: UUID, entity: T) -> T:
        """
        Update an existing entity.

        Args:
            entity_id: The ID of the entity to update
            entity: The updated entity

        Returns:
            The updated entity

        Raises:
            NotFoundException: If entity not found
        """
        with self._lock:
            if entity_id not in self._data:
                raise NotFoundException(
                    f"Entity with id {entity_id} not found",
                    {"id": str(entity_id)},
                )
            entity.updated_at = datetime.utcnow()
            self._data[entity_id] = entity
            return entity

    def delete(self, entity_id: UUID) -> None:
        """
        Delete an entity.

        Args:
            entity_id: The ID of the entity to delete

        Raises:
            NotFoundException: If entity not found
        """
        with self._lock:
            if entity_id not in self._data:
                raise NotFoundException(
                    f"Entity with id {entity_id} not found",
                    {"id": str(entity_id)},
                )
            del self._data[entity_id]

    def exists(self, entity_id: UUID) -> bool:
        """
        Check if an entity exists.

        Args:
            entity_id: The ID to check

        Returns:
            True if entity exists, False otherwise
        """
        with self._lock:
            return entity_id in self._data

    def clear(self) -> None:
        """Clear all data from the repository."""
        with self._lock:
            self._data.clear()


# Singleton instances
library_repository = InMemoryRepository[Library]()
document_repository = InMemoryRepository[Document]()
chunk_repository = InMemoryRepository[Chunk]()
