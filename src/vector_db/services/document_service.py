"""Service for document operations."""

from datetime import datetime
from uuid import UUID

from vector_db.core.exceptions import NotFoundException
from vector_db.models import (
    Chunk,
    Document,
    DocumentCreate,
    DocumentUpdate,
    PaginatedResponse,
)
from vector_db.repositories.registry import (
    get_chunk_repository,
    get_document_repository,
    get_library_repository,
)


class DocumentService:
    """Service for managing documents."""

    async def create_document(self, library_id: UUID, document_create: DocumentCreate) -> Document:
        """
        Create a new document in a library.

        Args:
            library_id: The parent library ID
            document_create: Document creation data

        Returns:
            The created document

        Raises:
            NotFoundException: If library not found
        """
        library_repo = get_library_repository()
        document_repo = get_document_repository()
        chunk_repo = get_chunk_repository()

        # Verify library exists
        library = library_repo.get(library_id)

        # Create the document first
        document = Document(
            library_id=library_id,
            name=document_create.name,
            metadata=document_create.metadata,
        )

        # Save document
        document = await document_repo.create(document)

        # Chunks are now created separately via the chunk API endpoint
        # No longer creating chunks inline since DocumentCreate doesn't have a chunks field

        # Update library timestamp
        library.updated_at = datetime.utcnow()
        await library_repo.update(library_id, library)

        return document

    async def get_document(self, document_id: UUID) -> Document:
        """
        Get a document by ID.

        Args:
            document_id: The document ID

        Returns:
            The document

        Raises:
            NotFoundException: If document not found
        """
        document_repo = get_document_repository()
        return await document_repo.get(document_id)

    async def get_all_documents(self, library_id: UUID) -> list[Document]:
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

    async def get_documents_paginated(
        self,
        library_id: UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> PaginatedResponse[Document]:
        """
        Get documents with pagination.

        Args:
            library_id: The library ID
            skip: Number of items to skip
            limit: Maximum number of items to return

        Returns:
            Paginated response with documents

        Raises:
            NotFoundException: If library not found
        """
        library_repo = get_library_repository()
        document_repo = get_document_repository()

        # Verify library exists
        library_repo.get(library_id)

        # Get documents directly from repository
        items, total = await document_repo.get_paginated_by_library(
            library_id, skip=skip, limit=limit
        )

        return PaginatedResponse(
            items=items,
            total=total,
            skip=skip,
            limit=limit,
            has_more=skip + len(items) < total,
        )

    async def update_document(
        self, document_id: UUID, document_update: DocumentUpdate
    ) -> Document:
        """
        Update a document.

        Args:
            document_id: The document ID
            document_update: Update data

        Returns:
            The updated document

        Raises:
            NotFoundException: If document not found
        """
        document_repo = get_document_repository()
        document = await document_repo.get(document_id)

        if document_update.name is not None:
            document.name = document_update.name
        if document_update.metadata is not None:
            document.metadata = document_update.metadata

        document.updated_at = datetime.utcnow()
        return await document_repo.update(document_id, document)

    async def delete_document(self, document_id: UUID) -> None:
        """
        Delete a document and all its chunks.

        Args:
            document_id: The document ID

        Raises:
            NotFoundException: If document not found
        """
        library_repo = get_library_repository()
        document_repo = get_document_repository()
        chunk_repo = get_chunk_repository()

        document = await document_repo.get(document_id)

        # Delete all chunks in the document
        chunks = await chunk_repo.get_by_document(document_id)
        for chunk in chunks:
            try:
                await chunk_repo.delete(chunk.id)
            except NotFoundException:
                pass

        # Update library timestamp
        library = library_repo.get(document.library_id)
        library.updated_at = datetime.utcnow()
        await library_repo.update(library.id, library)

        # Delete the document
        await document_repo.delete(document_id)

    async def get_chunks(self, document_id: UUID) -> list[Chunk]:
        """
        Get all chunks in a document.

        Args:
            document_id: The document ID

        Returns:
            List of chunks

        Raises:
            NotFoundException: If document not found
        """
        document_repo = get_document_repository()
        chunk_repo = get_chunk_repository()

        # Verify document exists
        await document_repo.get(document_id)

        # Get chunks from repository
        return await chunk_repo.get_by_document(document_id)


# Singleton instance
document_service = DocumentService()
