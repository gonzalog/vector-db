"""Service for library operations."""

from datetime import datetime
from uuid import UUID

from vector_db.core.exceptions import NotFoundException
from vector_db.models import (
    Document,
    Library,
    LibraryCreate,
    LibraryUpdate,
)
from vector_db.repositories.memory_repository import (
    document_repository,
    library_repository,
)


class LibraryService:
    """Service for managing libraries."""

    def create_library(self, library_create: LibraryCreate) -> Library:
        """
        Create a new library.

        Args:
            library_create: Library creation data

        Returns:
            The created library
        """
        library = Library(
            name=library_create.name,
            metadata=library_create.metadata,
            documents=[],
        )
        return library_repository.create(library)

    def get_library(self, library_id: UUID) -> Library:
        """
        Get a library by ID.

        Args:
            library_id: The library ID

        Returns:
            The library

        Raises:
            NotFoundException: If library not found
        """
        return library_repository.get(library_id)

    def get_all_libraries(self) -> list[Library]:
        """
        Get all libraries.

        Returns:
            List of all libraries
        """
        return library_repository.get_all()

    def update_library(
        self, library_id: UUID, library_update: LibraryUpdate
    ) -> Library:
        """
        Update a library.

        Args:
            library_id: The library ID
            library_update: Update data

        Returns:
            The updated library

        Raises:
            NotFoundException: If library not found
        """
        library = library_repository.get(library_id)

        if library_update.name is not None:
            library.name = library_update.name
        if library_update.metadata is not None:
            library.metadata = library_update.metadata

        library.updated_at = datetime.utcnow()
        return library_repository.update(library_id, library)

    def delete_library(self, library_id: UUID) -> None:
        """
        Delete a library and all its documents.

        Args:
            library_id: The library ID

        Raises:
            NotFoundException: If library not found
        """
        library = library_repository.get(library_id)

        # Delete all documents in the library
        for document in library.documents:
            try:
                document_repository.delete(document.id)
            except NotFoundException:
                pass

        library_repository.delete(library_id)

    def get_documents(self, library_id: UUID) -> list[Document]:
        """
        Get all documents in a library.

        Args:
            library_id: The library ID

        Returns:
            List of documents

        Raises:
            NotFoundException: If library not found
        """
        library = library_repository.get(library_id)
        return library.documents


# Singleton instance
library_service = LibraryService()
