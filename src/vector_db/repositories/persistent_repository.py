"""Persistent repository using SQLite + NumPy + Pickle."""

import numpy as np
from datetime import datetime
from enum import Enum
from threading import RLock
from uuid import UUID

from vector_db.core.exceptions import AlreadyExistsException, NotFoundException
from vector_db.core.locks import ReadWriteLock
from vector_db.core.persistence.database import get_database
from vector_db.core.persistence.repositories import (
    ChunkRepository,
    DocumentRepository,
    LibraryRepository as DBLibraryRepository,
)
from vector_db.core.persistence.vector_storage import VectorStorage
from vector_db.core.persistence.index_storage import IndexStorage
from vector_db.core.settings import Settings, settings as default_settings
from vector_db.indexes import DistanceMetric, FlatIndex, HNSWIndex, LSHIndex, VectorIndex
from vector_db.models import Chunk, Document, Library, SearchResponse, SearchResult, VectorQuery


class IndexType(str, Enum):
    """Supported index types."""

    FLAT = "flat"
    LSH = "lsh"
    HNSW = "hnsw"


class PersistentLibraryRepository:
    """
    Persistent repository for libraries with integrated index management.

    Combines SQLite (metadata), NumPy (.npy for vectors), and Pickle (.pkl for indexes).
    """

    def __init__(self, settings: Settings | None = None):
        """Initialize the persistent library repository."""
        if settings is None:
            settings = default_settings

        # Storage managers
        self.vector_storage = VectorStorage(settings.VECTORS_DIR)
        self.index_storage = IndexStorage(settings.INDEXES_DIR)

        # In-memory caches for performance
        self._libraries: dict[UUID, Library] = {}
        self._indexes: dict[UUID, VectorIndex] = {}
        self._vectors: dict[UUID, np.ndarray] = {}  # Cache vectors in memory

        # Locking (same strategy as in-memory repo)
        self._global_lock = RLock()
        self._library_locks: dict[UUID, ReadWriteLock] = {}

        # Track vector indices for chunks
        self._chunk_vector_indices: dict[UUID, int] = {}  # chunk_id -> vector_index

    async def initialize(self) -> None:
        """Load all libraries from database on startup."""
        db = get_database()
        lib_repo = DBLibraryRepository(db)

        libraries = await lib_repo.list(skip=0, limit=10000)  # Load all

        for library in libraries:
            with self._global_lock:
                self._libraries[library.id] = library
                self._library_locks[library.id] = ReadWriteLock()

                # Load vectors
                vectors = self.vector_storage.load(str(library.id))
                if vectors is not None:
                    self._vectors[library.id] = vectors

                # Try to load pickled index
                index = self.index_storage.load(str(library.id))

                if index is None:
                    # Index file missing or corrupted - rebuild from vectors
                    if vectors is not None:
                        index = await self._rebuild_index(library, vectors)
                    else:
                        # No vectors either - create empty index
                        index = self._create_empty_index(library)

                self._indexes[library.id] = index

    async def _rebuild_index(self, library: Library, vectors: np.ndarray) -> VectorIndex:
        """Rebuild index from vectors and chunk metadata."""
        # Create new empty index
        index = self._create_empty_index(library)

        # Load all chunks for this library
        db = get_database()
        chunk_repo = ChunkRepository(db)
        chunks_with_indices = await chunk_repo.list_by_library(str(library.id))

        # Add chunks back to index
        for chunk, vector_index in chunks_with_indices:
            # Set the embedding from vectors array
            if vector_index < len(vectors):
                chunk_with_embedding = Chunk(
                    id=chunk.id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    embedding=vectors[vector_index].tolist(),
                    metadata=chunk.metadata,
                    created_at=chunk.created_at,
                )
                index.add(chunk_with_embedding)
                self._chunk_vector_indices[chunk.id] = vector_index

        # Save rebuilt index
        self.index_storage.save(str(library.id), index)

        return index

    def _create_empty_index(self, library: Library) -> VectorIndex:
        """Create an empty index for a library based on its configuration."""
        index_type = IndexType(library.index_config.index_type)
        distance_metric = DistanceMetric(library.index_config.distance_metric)

        if index_type == IndexType.FLAT:
            return FlatIndex(distance_metric=distance_metric)
        elif index_type == IndexType.LSH:
            n_hash_tables = library.index_config.n_hash_tables or 5
            n_hash_bits = library.index_config.n_hash_bits or 8
            return LSHIndex(
                n_hash_tables=n_hash_tables,
                n_hash_bits=n_hash_bits,
                distance_metric=distance_metric,
            )
        elif index_type == IndexType.HNSW:
            M = library.index_config.M or 16
            ef_construction = library.index_config.ef_construction or 200
            ef_search = library.index_config.ef_search or 50
            return HNSWIndex(
                M=M,
                ef_construction=ef_construction,
                ef_search=ef_search,
                distance_metric=distance_metric,
            )
        else:
            raise ValueError(f"Unknown index type: {index_type}")

    async def create(self, entity: Library) -> Library:
        """Create a new library."""
        with self._global_lock:
            # Check if exists
            if entity.id in self._libraries:
                raise AlreadyExistsException(
                    f"Entity with id {entity.id} already exists",
                    {"id": str(entity.id)},
                )

            # Save to database
            db = get_database()
            lib_repo = DBLibraryRepository(db)
            await lib_repo.create(entity)

            # Cache in memory
            self._libraries[entity.id] = entity
            self._library_locks[entity.id] = ReadWriteLock()

            # Create empty index and empty vectors array
            index = self._create_empty_index(entity)
            self._indexes[entity.id] = index
            self._vectors[entity.id] = np.empty((0, 0), dtype=np.float32)

            # Save empty structures
            self.vector_storage.save(str(entity.id), self._vectors[entity.id])
            self.index_storage.save(str(entity.id), index)

            return entity

    def get(self, entity_id: UUID) -> Library:
        """Get a library by ID."""
        with self._global_lock:
            if entity_id not in self._libraries:
                raise NotFoundException(
                    f"Entity with id {entity_id} not found",
                    {"id": str(entity_id)},
                )
            return self._libraries[entity_id]

    def get_all(self) -> list[Library]:
        """Get all libraries."""
        with self._global_lock:
            return list(self._libraries.values())

    def get_paginated(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> tuple[list[Library], int]:
        """Get libraries with pagination."""
        with self._global_lock:
            items = list(self._libraries.values())
            total = len(items)
            paginated_items = items[skip : skip + limit]
            return paginated_items, total

    async def delete(self, entity_id: UUID) -> None:
        """Delete a library and all associated data."""
        with self._global_lock:
            if entity_id not in self._libraries:
                raise NotFoundException(
                    f"Entity with id {entity_id} not found",
                    {"id": str(entity_id)},
                )

            # Delete from database (cascades to documents and chunks)
            db = get_database()
            lib_repo = DBLibraryRepository(db)
            await lib_repo.delete(str(entity_id))

            # Delete files
            self.vector_storage.delete(str(entity_id))
            self.index_storage.delete(str(entity_id))

            # Remove from memory
            del self._libraries[entity_id]
            del self._indexes[entity_id]
            del self._vectors[entity_id]
            if entity_id in self._library_locks:
                del self._library_locks[entity_id]

    def get_index(self, library_id: UUID) -> VectorIndex | None:
        """Get the index for a library."""
        if library_id not in self._library_locks:
            return None

        with self._library_locks[library_id].read():
            return self._indexes.get(library_id)

    async def add_chunk_to_index(self, library_id: UUID, chunk: Chunk) -> None:
        """Add a chunk to the library's index."""
        # Verify library exists
        self.get(library_id)

        if library_id not in self._library_locks:
            return

        with self._library_locks[library_id].write():
            index = self._indexes.get(library_id)
            if not index:
                return

            # Get current vectors array
            vectors = self._vectors.get(library_id)
            if vectors is None:
                return

            # Append new embedding to vectors array
            embedding_array = np.array([chunk.embedding], dtype=np.float32)
            if vectors.size == 0:
                # First vector
                new_vectors = embedding_array
            else:
                new_vectors = np.vstack([vectors, embedding_array])

            vector_index = len(new_vectors) - 1

            # Update cache
            self._vectors[library_id] = new_vectors
            self._chunk_vector_indices[chunk.id] = vector_index

            # Add to index
            index.add(chunk)

            # Save to disk
            self.vector_storage.save(str(library_id), new_vectors)
            self.index_storage.save(str(library_id), index)

    async def remove_chunk_from_index(self, library_id: UUID, chunk_id: UUID) -> bool:
        """Remove a chunk from the library's index."""
        if library_id not in self._library_locks:
            return False

        with self._library_locks[library_id].write():
            index = self._indexes.get(library_id)
            if not index:
                return False

            # Remove from index
            removed = index.remove(chunk_id)

            if removed:
                # Note: We don't remove from vectors array to maintain indices
                # Just mark as removed in index
                #  In production, you'd want periodic compaction

                # Save updated index
                self.index_storage.save(str(library_id), index)

            return removed

    def search(self, library_id: UUID, query: VectorQuery) -> SearchResponse:
        """Search for similar vectors in a library."""
        # Verify library exists
        self.get(library_id)

        if library_id not in self._library_locks:
            raise NotFoundException(
                f"Library with id {library_id} not found",
                {"id": str(library_id)},
            )

        with self._library_locks[library_id].read():
            index = self._indexes.get(library_id)
            if index is None:
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
                    score=1.0 / (1.0 + distance),
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

    def exists(self, entity_id: UUID) -> bool:
        """Check if a library exists."""
        with self._global_lock:
            return entity_id in self._libraries
