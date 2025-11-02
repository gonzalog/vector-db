"""Persistent repository for documents."""

from threading import RLock
from uuid import UUID

from vector_db.core.exceptions import AlreadyExistsException, NotFoundException
from vector_db.core.persistence.database import get_database
from vector_db.core.persistence.repositories import (
    DocumentRepository as DBDocumentRepository,
)
from vector_db.models import Document


from threading import RLock
from uuid import UUID

from vector_db.core.exceptions import AlreadyExistsException, NotFoundException
from vector_db.core.persistence.database import get_database
from vector_db.core.persistence.repositories import (
    ChunkRepository as DBChunkRepository,
    DocumentRepository as DBDocumentRepository,
)
from vector_db.models import Chunk, Document


class PersistentDocumentRepository:
    """Persistent repository for documents using SQLite."""

    def __init__(self):
        """Initialize the repository."""
        self._lock = RLock()
        self._cache: dict[UUID, Document] = {}

    async def initialize(self) -> None:
        """Load all documents into cache."""
        # Documents are loaded on-demand, no need to load all upfront
        pass

    async def create(self, entity: Document) -> Document:
        """Create a new document."""
        with self._lock:
            if entity.id in self._cache:
                raise AlreadyExistsException(
                    f"Entity with id {entity.id} already exists",
                    {"id": str(entity.id)},
                )

            db = get_database()
            doc_repo = DBDocumentRepository(db)
            await doc_repo.create(entity)

            self._cache[entity.id] = entity
            return entity

    async def get(self, entity_id: UUID) -> Document:
        """Get a document by ID."""
        with self._lock:
            # Check cache first
            if entity_id in self._cache:
                return self._cache[entity_id]

            # Load from database
            db = get_database()
            doc_repo = DBDocumentRepository(db)
            document = await doc_repo.get(str(entity_id))

            if not document:
                raise NotFoundException(
                    f"Entity with id {entity_id} not found",
                    {"id": str(entity_id)},
                )

            self._cache[entity_id] = document
            return document

    async def get_all(self) -> list[Document]:
        """Get all documents (not recommended for large datasets)."""
        db = get_database()
        doc_repo = DBDocumentRepository(db)
        # This would need to be paginated for real use
        return await doc_repo.list_by_library("", skip=0, limit=100000)

    async def get_paginated_by_library(
        self, library_id: UUID, skip: int = 0, limit: int = 100
    ) -> tuple[list[Document], int]:
        """Get documents in a library with pagination."""
        db = get_database()
        doc_repo = DBDocumentRepository(db)
        documents = await doc_repo.list_by_library(str(library_id), skip, limit)
        total = await doc_repo.count_by_library(str(library_id))

        # Cache them
        with self._lock:
            for doc in documents:
                self._cache[doc.id] = doc

        return documents, total

    async def update(self, entity_id: UUID, entity: Document) -> Document:
        """Update a document."""
        with self._lock:
            db = get_database()
            doc_repo = DBDocumentRepository(db)
            await doc_repo.update(str(entity_id), entity)

            self._cache[entity_id] = entity
            return entity

    def get_paginated(self, skip: int = 0, limit: int = 100, filter_fn=None) -> tuple[list[Document], int]:
        """Get documents with optional filtering (operates on cache)."""
        with self._lock:
            all_docs = list(self._cache.values())

            if filter_fn:
                filtered = [doc for doc in all_docs if filter_fn(doc)]
            else:
                filtered = all_docs

            total = len(filtered)
            items = filtered[skip:skip + limit]
            return items, total

    async def delete(self, entity_id: UUID) -> None:
        """Delete a document."""
        with self._lock:
            db = get_database()
            doc_repo = DBDocumentRepository(db)
            await doc_repo.delete(str(entity_id))

            if entity_id in self._cache:
                del self._cache[entity_id]

    def exists(self, entity_id: UUID) -> bool:
        """Check if a document exists."""
        with self._lock:
            return entity_id in self._cache
