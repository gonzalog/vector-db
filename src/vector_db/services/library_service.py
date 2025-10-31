"""Service for library operations."""

from datetime import datetime
from uuid import UUID

from vector_db.core.exceptions import NotFoundException
from vector_db.models import (
    Document,
    Library,
    LibraryCreate,
    LibraryUpdate,
    PaginatedResponse,
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
            index_config=library_create.index_config,
            documents=[],
        )
        # Repository handles index creation
        library = library_repository.create(library)
        return library

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

    def get_libraries_paginated(
        self, skip: int = 0, limit: int = 100
    ) -> PaginatedResponse[Library]:
        """
        Get libraries with pagination.

        Args:
            skip: Number of items to skip
            limit: Maximum number of items to return

        Returns:
            Paginated response with libraries
        """
        items, total = library_repository.get_paginated(
            skip=skip, limit=limit
        )

        return PaginatedResponse(
            items=items,
            total=total,
            skip=skip,
            limit=limit,
            has_more=skip + len(items) < total,
        )

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

        # Repository handles index deletion
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
