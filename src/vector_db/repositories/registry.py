"""Repository registry for managing singleton persistent repositories."""

from vector_db.repositories.persistent_repository import PersistentLibraryRepository
from vector_db.repositories.persistent_document_chunk import (
    PersistentDocumentRepository,
    PersistentChunkRepository,
)


# Singleton instances (will be initialized on app startup)
_library_repository: PersistentLibraryRepository | None = None
_document_repository: PersistentDocumentRepository | None = None
_chunk_repository: PersistentChunkRepository | None = None


async def initialize_repositories() -> None:
    """Initialize all repositories. Call this on app startup."""
    global _library_repository, _document_repository, _chunk_repository

    # Initialize all persistent repositories
    _library_repository = PersistentLibraryRepository()
    await _library_repository.initialize()

    _document_repository = PersistentDocumentRepository()
    await _document_repository.initialize()

    _chunk_repository = PersistentChunkRepository()
    await _chunk_repository.initialize()


def get_library_repository() -> PersistentLibraryRepository:
    """Get the library repository singleton."""
    if _library_repository is None:
        raise RuntimeError("Repositories not initialized. Call initialize_repositories() first.")
    return _library_repository


def get_document_repository() -> PersistentDocumentRepository:
    """Get the document repository singleton."""
    if _document_repository is None:
        raise RuntimeError("Repositories not initialized. Call initialize_repositories() first.")
    return _document_repository


def get_chunk_repository() -> PersistentChunkRepository:
    """Get the chunk repository singleton."""
    if _chunk_repository is None:
        raise RuntimeError("Repositories not initialized. Call initialize_repositories() first.")
    return _chunk_repository
