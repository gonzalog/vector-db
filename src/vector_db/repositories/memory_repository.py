"""In-memory repository for storing vector database data."""

from datetime import datetime
from enum import Enum
from threading import RLock
from typing import Callable, Generic, TypeVar
from uuid import UUID

from vector_db.core.exceptions import AlreadyExistsException, NotFoundException
from vector_db.core.locks import ReadWriteLock
from vector_db.indexes import DistanceMetric, FlatIndex, HNSWIndex, LSHIndex, VectorIndex
from vector_db.models import Chunk, Document, Library, SearchResponse, SearchResult, VectorQuery

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

    def get_paginated(
        self,
        skip: int = 0,
        limit: int = 100,
        filter_fn: Callable[[T], bool] | None = None,
    ) -> tuple[list[T], int]:
        """
        Get entities with pagination and optional filtering.

        Args:
            skip: Number of items to skip
            limit: Maximum number of items to return
            filter_fn: Optional filter function

        Returns:
            Tuple of (items, total_count)
        """
        with self._lock:
            items = list(self._data.values())

            # Apply filter if provided
            if filter_fn:
                items = [item for item in items if filter_fn(item)]

            total = len(items)

            # Apply pagination
            paginated_items = items[skip : skip + limit]

            return paginated_items, total

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


class IndexType(str, Enum):
    """Supported index types."""

    FLAT = "flat"
    LSH = "lsh"
    HNSW = "hnsw"


class LibraryRepository(InMemoryRepository[Library]):
    """
    Repository for libraries with integrated index management.

    Uses a two-level locking strategy:
    - Global lock (_global_lock): For library creation and deletion
    - Per-library RWLock: For operations on individual libraries
        - Read lock: For search operations (allows concurrent reads)
        - Write lock: For add/remove chunk operations (exclusive access)
    """

    def __init__(self):
        """Initialize the library repository with index storage and locks."""
        super().__init__()
        self._indexes: dict[UUID, VectorIndex] = {}

        # Global lock for library creation/deletion
        self._global_lock = RLock()

        # Per-library ReadWriteLocks for concurrent read operations
        self._library_locks: dict[UUID, ReadWriteLock] = {}

    def create(self, entity: Library) -> Library:
        """
        Create a new library and its index.

        Uses global lock to ensure thread-safe creation.

        Args:
            entity: The library to create

        Returns:
            The created library

        Raises:
            AlreadyExistsException: If library with same ID already exists
        """
        with self._global_lock:
            library = super().create(entity)
            # Create index and RWLock for this library
            self._create_index_for_library(library)
            self._library_locks[library.id] = ReadWriteLock()
            return library

    def delete(self, entity_id: UUID) -> None:
        """
        Delete a library and its index.

        Uses global lock to ensure thread-safe deletion.

        Args:
            entity_id: The ID of the library to delete

        Raises:
            NotFoundException: If library not found
        """
        with self._global_lock:
            super().delete(entity_id)
            self._delete_index_for_library(entity_id)
            # Remove the RWLock for this library
            if entity_id in self._library_locks:
                del self._library_locks[entity_id]

    def clear(self) -> None:
        """Clear all libraries and indexes."""
        with self._global_lock:
            super().clear()
            self._indexes.clear()
            self._library_locks.clear()

    def _create_index_for_library(self, library: Library) -> VectorIndex:
        """
        Create an index for a library based on its configuration.

        Note: Called from create() which already holds global lock.

        Args:
            library: The library to create an index for

        Returns:
            The created index
        """
        index_type = IndexType(library.index_config.index_type)
        distance_metric = DistanceMetric(library.index_config.distance_metric)

        # Create the appropriate index type
        if index_type == IndexType.FLAT:
            index = FlatIndex(distance_metric=distance_metric)
        elif index_type == IndexType.LSH:
            n_hash_tables = library.index_config.n_hash_tables or 5
            n_hash_bits = library.index_config.n_hash_bits or 8
            index = LSHIndex(
                n_hash_tables=n_hash_tables,
                n_hash_bits=n_hash_bits,
                distance_metric=distance_metric,
            )
        elif index_type == IndexType.HNSW:
            M = library.index_config.M or 16
            ef_construction = library.index_config.ef_construction or 200
            ef_search = library.index_config.ef_search or 50
            index = HNSWIndex(
                M=M,
                ef_construction=ef_construction,
                ef_search=ef_search,
                distance_metric=distance_metric,
            )
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        self._indexes[library.id] = index
        return index

    def _delete_index_for_library(self, library_id: UUID) -> bool:
        """
        Delete the index for a library.

        Note: Called from delete() which already holds global lock.

        Args:
            library_id: ID of the library

        Returns:
            True if deleted, False if not found
        """
        if library_id in self._indexes:
            del self._indexes[library_id]
            return True
        return False

    def get_index(self, library_id: UUID) -> VectorIndex | None:
        """
        Get the index for a library.

        Uses read lock to allow concurrent access.

        Args:
            library_id: ID of the library

        Returns:
            The index if it exists, None otherwise
        """
        if library_id not in self._library_locks:
            return None

        with self._library_locks[library_id].read():
            return self._indexes.get(library_id)

    def add_chunk_to_index(self, library_id: UUID, chunk: Chunk) -> None:
        """
        Add a chunk to the library's index.

        Uses write lock for exclusive access during modification.

        Args:
            library_id: ID of the library
            chunk: Chunk to add

        Raises:
            NotFoundException: If library doesn't exist
        """
        # Verify library exists
        self.get(library_id)

        if library_id not in self._library_locks:
            return

        # Use write lock for exclusive access
        with self._library_locks[library_id].write():
            index = self._indexes.get(library_id)
            if index:
                index.add(chunk)

    def remove_chunk_from_index(self, library_id: UUID, chunk_id: UUID) -> bool:
        """
        Remove a chunk from the library's index.

        Uses write lock for exclusive access during modification.

        Args:
            library_id: ID of the library
            chunk_id: ID of the chunk to remove

        Returns:
            True if removed, False if not found
        """
        if library_id not in self._library_locks:
            return False

        # Use write lock for exclusive access
        with self._library_locks[library_id].write():
            index = self._indexes.get(library_id)
            if index:
                return index.remove(chunk_id)
            return False

    def update_chunk_in_index(self, library_id: UUID, chunk: Chunk) -> bool:
        """
        Update a chunk in the library's index.

        Uses write lock for exclusive access during modification.

        Args:
            library_id: ID of the library
            chunk: Updated chunk with new embedding

        Returns:
            True if updated, False if not found
        """
        # Verify library exists
        self.get(library_id)

        if library_id not in self._library_locks:
            return False

        # Use write lock for exclusive access
        with self._library_locks[library_id].write():
            index = self._indexes.get(library_id)
            if index:
                # Remove old chunk and add updated chunk
                index.remove(chunk.id)
                index.add(chunk)
                return True
            return False

    def search(self, library_id: UUID, query: VectorQuery) -> SearchResponse:
        """
        Search for similar vectors in a library.

        Uses read lock to allow multiple concurrent searches.

        Args:
            library_id: ID of the library to search
            query: Vector query with embedding and filters

        Returns:
            Search response with results

        Raises:
            NotFoundException: If library doesn't exist
        """
        # Verify library exists
        self.get(library_id)

        if library_id not in self._library_locks:
            # Library was just deleted
            raise NotFoundException(
                f"Library with id {library_id} not found",
                {"id": str(library_id)},
            )

        # Use read lock to allow concurrent searches
        with self._library_locks[library_id].read():
            index = self._indexes.get(library_id)
            if index is None:
                # Return empty results if no index
                return SearchResponse(
                    query=query,
                    results=[],
                    total_results=0,
                    library_id=library_id,
                )

            # Perform search
            results = index.search(
                query_embedding=query.embedding,
                k=query.k,
                metadata_filter=query.metadata_filter,
            )

            # Convert to SearchResult objects
            search_results = [
                SearchResult(
                    chunk=chunk,
                    score=1.0 / (1.0 + distance),  # Convert distance to similarity score
                    distance=distance,
                )
                for chunk, distance in results
            ]

            return SearchResponse(
                query=query,
                results=search_results,
                total_results=len(search_results),
                library_id=library_id,
            )


# Singleton instances
library_repository = LibraryRepository()
document_repository = InMemoryRepository[Document]()
chunk_repository = InMemoryRepository[Chunk]()
