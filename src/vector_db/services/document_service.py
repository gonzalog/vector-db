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
from vector_db.repositories.memory_repository import (
    chunk_repository,
    document_repository,
    library_repository,
)


class DocumentService:
    """Service for managing documents."""

    def create_document(self, library_id: UUID, document_create: DocumentCreate) -> Document:
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
        # Verify library exists
        library = library_repository.get(library_id)

        # Create the document
        document = Document(
            library_id=library_id,
            name=document_create.name,
            metadata=document_create.metadata,
            chunks=[],
        )

        # Create chunks if provided
        for chunk_create in document_create.chunks:
            chunk = Chunk(
                document_id=document.id,
                text=chunk_create.text,
                embedding=chunk_create.embedding,
                metadata=chunk_create.metadata,
            )
            chunk = chunk_repository.create(chunk)
            document.chunks.append(chunk)

        # Save document
        document = document_repository.create(document)

        # Add document to library
        library.documents.append(document)
        library.updated_at = datetime.utcnow()
        library_repository.update(library_id, library)

        return document

    def get_document(self, document_id: UUID) -> Document:
        """
        Get a document by ID.

        Args:
            document_id: The document ID

        Returns:
            The document

        Raises:
            NotFoundException: If document not found
        """
        return document_repository.get(document_id)

    def get_all_documents(self, library_id: UUID) -> list[Document]:
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

    def get_documents_paginated(
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
        # Verify library exists
        library_repository.get(library_id)

        def filter_fn(document: Document) -> bool:
            return document.library_id == library_id

        items, total = document_repository.get_paginated(
            skip=skip, limit=limit, filter_fn=filter_fn
        )

        return PaginatedResponse(
            items=items,
            total=total,
            skip=skip,
            limit=limit,
            has_more=skip + len(items) < total,
        )

    def update_document(
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
        document = document_repository.get(document_id)

        if document_update.name is not None:
            document.name = document_update.name
        if document_update.metadata is not None:
            document.metadata = document_update.metadata

        document.updated_at = datetime.utcnow()
        return document_repository.update(document_id, document)

    def delete_document(self, document_id: UUID) -> None:
        """
        Delete a document and all its chunks.

        Args:
            document_id: The document ID

        Raises:
            NotFoundException: If document not found
        """
        document = document_repository.get(document_id)

        # Delete all chunks in the document
        for chunk in document.chunks:
            try:
                chunk_repository.delete(chunk.id)
            except NotFoundException:
                pass

        # Remove document from library
        library = library_repository.get(document.library_id)
        library.documents = [d for d in library.documents if d.id != document_id]
        library.updated_at = datetime.utcnow()
        library_repository.update(library.id, library)

        # Delete the document
        document_repository.delete(document_id)

    def get_chunks(self, document_id: UUID) -> list[Chunk]:
        """
        Get all chunks in a document.

        Args:
            document_id: The document ID

        Returns:
            List of chunks

        Raises:
            NotFoundException: If document not found
        """
        document = document_repository.get(document_id)
        return document.chunks


# Singleton instance
document_service = DocumentService()
