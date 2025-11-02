"""Persistent repository for chunks."""

from threading import RLock
from uuid import UUID

from vector_db.core.exceptions import AlreadyExistsException, NotFoundException
from vector_db.core.persistence.database import get_database
from vector_db.core.persistence.repositories import (
    ChunkRepository as DBChunkRepository,
)
from vector_db.models import Chunk


class PersistentChunkRepository:
    """Persistent repository for chunks using SQLite."""

    def __init__(self):
        """Initialize the repository."""
        self._lock = RLock()
        self._cache: dict[UUID, Chunk] = {}

    async def initialize(self) -> None:
        """Load chunks into cache (on-demand)."""
        pass

    async def create(self, entity: Chunk, vector_index: int) -> Chunk:
        """Create a new chunk."""
        with self._lock:
            if entity.id in self._cache:
                raise AlreadyExistsException(
                    f"Entity with id {entity.id} already exists",
                    {"id": str(entity.id)},
                )

            db = get_database()
            chunk_repo = DBChunkRepository(db)
            await chunk_repo.create(entity, vector_index)

            self._cache[entity.id] = entity
            return entity

    async def get(self, entity_id: UUID) -> Chunk:
        """Get a chunk by ID."""
        with self._lock:
            # Check cache
            if entity_id in self._cache:
                return self._cache[entity_id]

            # Load from database
            db = get_database()
            chunk_repo = DBChunkRepository(db)
            result = await chunk_repo.get(str(entity_id))

            if not result:
                raise NotFoundException(
                    f"Entity with id {entity_id} not found",
                    {"id": str(entity_id)},
                )

            chunk, _ = result
            self._cache[entity_id] = chunk
            return chunk

    async def get_by_document(self, document_id: UUID) -> list[Chunk]:
        """Get all chunks in a document."""
        db = get_database()
        chunk_repo = DBChunkRepository(db)
        chunks_with_indices = await chunk_repo.list_by_document(str(document_id))

        chunks = [chunk for chunk, _ in chunks_with_indices]

        # Cache them
        with self._lock:
            for chunk in chunks:
                self._cache[chunk.id] = chunk

        return chunks

    async def update(self, entity_id: UUID, entity: Chunk) -> Chunk:
        """Update a chunk."""
        with self._lock:
            db = get_database()
            chunk_repo = DBChunkRepository(db)
            await chunk_repo.update(str(entity_id), entity)

            self._cache[entity_id] = entity
            return entity

    def get_paginated(self, skip: int = 0, limit: int = 100, filter_fn=None) -> tuple[list[Chunk], int]:
        """Get chunks with optional filtering (operates on cache)."""
        with self._lock:
            all_chunks = list(self._cache.values())

            if filter_fn:
                filtered = [chunk for chunk in all_chunks if filter_fn(chunk)]
            else:
                filtered = all_chunks

            total = len(filtered)
            items = filtered[skip:skip + limit]
            return items, total

    async def delete(self, entity_id: UUID) -> None:
        """Delete a chunk."""
        with self._lock:
            db = get_database()
            chunk_repo = DBChunkRepository(db)
            await chunk_repo.delete(str(entity_id))

            if entity_id in self._cache:
                del self._cache[entity_id]

    def exists(self, entity_id: UUID) -> bool:
        """Check if a chunk exists."""
        with self._lock:
            return entity_id in self._cache
