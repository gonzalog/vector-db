"""Repository registry for managing singleton persistent repositories."""

from vector_db.repositories.persistent_repository import PersistentLibraryRepository
from vector_db.repositories.memory_repository import InMemoryRepository
from vector_db.models import Document, Chunk


# Singleton instances (will be initialized on app startup)
_library_repository: PersistentLibraryRepository | None = None
_document_repository: InMemoryRepository[Document] | None = None
_chunk_repository: InMemoryRepository[Chunk] | None = None


async def initialize_repositories() -> None:
    """Initialize all repositories. Call this on app startup."""
    global _library_repository, _document_repository, _chunk_repository

    # Initialize persistent library repository
    _library_repository = PersistentLibraryRepository()
    await _library_repository.initialize()

    # For now, keep documents and chunks in memory
    # TODO: Create persistent versions of these too
    _document_repository = InMemoryRepository[Document]()
    _chunk_repository = InMemoryRepository[Chunk]()


def get_library_repository() -> PersistentLibraryRepository:
    """Get the library repository singleton."""
    if _library_repository is None:
        raise RuntimeError("Repositories not initialized. Call initialize_repositories() first.")
    return _library_repository


def get_document_repository() -> InMemoryRepository[Document]:
    """Get the document repository singleton."""
    if _document_repository is None:
        raise RuntimeError("Repositories not initialized. Call initialize_repositories() first.")
    return _document_repository


def get_chunk_repository() -> InMemoryRepository[Chunk]:
    """Get the chunk repository singleton."""
    if _chunk_repository is None:
        raise RuntimeError("Repositories not initialized. Call initialize_repositories() first.")
    return _chunk_repository
