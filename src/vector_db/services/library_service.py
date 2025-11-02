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
    SearchResponse,
    VectorQuery,
)
from vector_db.repositories.registry import (
    get_library_repository,
    get_document_repository,
)


class LibraryService:
    """Service for managing libraries."""

    async def create_library(self, library_create: LibraryCreate) -> Library:
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
        )
        # Repository handles index creation
        library_repo = get_library_repository()
        library = await library_repo.create(library)
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
        library_repo = get_library_repository()
        return library_repo.get(library_id)

    def get_all_libraries(self) -> list[Library]:
        """
        Get all libraries.

        Returns:
            List of all libraries
        """
        library_repo = get_library_repository()
        return library_repo.get_all()

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
        library_repo = get_library_repository()
        items, total = library_repo.get_paginated(
            skip=skip, limit=limit
        )

        return PaginatedResponse(
            items=items,
            total=total,
            skip=skip,
            limit=limit,
            has_more=skip + len(items) < total,
        )

    async def update_library(
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
        library_repo = get_library_repository()
        library = library_repo.get(library_id)

        if library_update.name is not None:
            library.name = library_update.name
        if library_update.metadata is not None:
            library.metadata = library_update.metadata

        library.updated_at = datetime.utcnow()
        return await library_repo.update(library_id, library)

    async def delete_library(self, library_id: UUID) -> None:
        """
        Delete a library and all its documents.

        Args:
            library_id: The library ID

        Raises:
            NotFoundException: If library not found
        """
        library_repo = get_library_repository()
        document_repo = get_document_repository()

        library = library_repo.get(library_id)

        # Delete all documents in the library
        documents, _ = await document_repo.get_paginated_by_library(library_id, 0, 10000)
        for document in documents:
            try:
                await document_repo.delete(document.id)
            except NotFoundException:
                pass

        # Repository handles index deletion
        await library_repo.delete(library_id)

    async def get_documents(self, library_id: UUID) -> list[Document]:
        """
        Get all documents in a library.

        Args:
            library_id: The library ID

        Returns:
            List of documents

        Raises:
            NotFoundException: If library not found
        """
        library_repo = get_library_repository()
        document_repo = get_document_repository()

        # Verify library exists
        library_repo.get(library_id)

        # Get documents from repository
        documents, _ = await document_repo.get_paginated_by_library(library_id, 0, 10000)
        return documents

    def search_library(self, library_id: UUID, query: VectorQuery) -> SearchResponse:
        """
        Search for similar vectors in a library.

        Args:
            library_id: The library ID
            query: Vector query with embedding and filters

        Returns:
            Search response with results

        Raises:
            NotFoundException: If library not found
        """
        library_repo = get_library_repository()
        return library_repo.search(library_id, query)


# Singleton instance
library_service = LibraryService()
