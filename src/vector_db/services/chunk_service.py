"""Service for chunk operations."""

from datetime import datetime
from uuid import UUID

from vector_db.core.exceptions import NotFoundException
from vector_db.models import Chunk, ChunkCreate, ChunkUpdate, PaginatedResponse
from vector_db.repositories.registry import (
    get_chunk_repository,
    get_document_repository,
    get_library_repository,
)


class ChunkService:
    """Service for managing chunks."""

    async def create_chunk(self, document_id: UUID, chunk_create: ChunkCreate) -> Chunk:
        """
        Create a new chunk in a document.

        Args:
            document_id: The parent document ID
            chunk_create: Chunk creation data

        Returns:
            The created chunk

        Raises:
            NotFoundException: If document not found
        """
        document_repo = get_document_repository()
        chunk_repo = get_chunk_repository()
        library_repo = get_library_repository()

        # Verify document exists
        document = await document_repo.get(document_id)

        # Create the chunk
        chunk = Chunk(
            document_id=document_id,
            text=chunk_create.text,
            embedding=chunk_create.embedding,
            metadata=chunk_create.metadata,
        )

        # Get the library to determine vector index
        library = library_repo.get(document.library_id)

        # Determine the next vector index for this library
        # For now, use a simple approach - count existing chunks
        # TODO: This could be optimized with a separate counter
        vector_index = 0  # Will be set by the repository

        # Save chunk with vector index
        chunk = await chunk_repo.create(chunk, vector_index)

        # Update document timestamp
        document.updated_at = datetime.utcnow()
        await document_repo.update(document_id, document)

        # Add to index
        try:
            await library_repo.add_chunk_to_index(document.library_id, chunk)
        except Exception:
            # Index errors shouldn't fail chunk creation
            pass

        return chunk

    async def get_chunk(self, chunk_id: UUID) -> Chunk:
        """
        Get a chunk by ID.

        Args:
            chunk_id: The chunk ID

        Returns:
            The chunk

        Raises:
            NotFoundException: If chunk not found
        """
        chunk_repo = get_chunk_repository()
        return await chunk_repo.get(chunk_id)

    async def get_all_chunks(self, document_id: UUID) -> list[Chunk]:
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

    async def get_chunks_paginated(
        self,
        document_id: UUID,
        skip: int = 0,
        limit: int = 100,
        search: str | None = None,
    ) -> PaginatedResponse[Chunk]:
        """
        Get chunks with pagination and optional search filtering.

        Args:
            document_id: The document ID
            skip: Number of items to skip
            limit: Maximum number of items to return
            search: Optional search term to filter by text

        Returns:
            Paginated response with chunks

        Raises:
            NotFoundException: If document not found
        """
        document_repo = get_document_repository()
        chunk_repo = get_chunk_repository()

        # Verify document exists
        await document_repo.get(document_id)

        def filter_fn(chunk: Chunk) -> bool:
            if chunk.document_id != document_id:
                return False
            if not search:
                return True
            return search.lower() in chunk.text.lower()

        items, total = chunk_repo.get_paginated(
            skip=skip, limit=limit, filter_fn=filter_fn
        )

        return PaginatedResponse(
            items=items,
            total=total,
            skip=skip,
            limit=limit,
            has_more=skip + len(items) < total,
        )

    async def update_chunk(self, chunk_id: UUID, chunk_update: ChunkUpdate) -> Chunk:
        """
        Update a chunk.

        Args:
            chunk_id: The chunk ID
            chunk_update: Update data

        Returns:
            The updated chunk

        Raises:
            NotFoundException: If chunk not found
        """
        chunk_repo = get_chunk_repository()
        document_repo = get_document_repository()
        library_repo = get_library_repository()

        chunk = await chunk_repo.get(chunk_id)

        if chunk_update.text is not None:
            chunk.text = chunk_update.text
        if chunk_update.embedding is not None:
            chunk.embedding = chunk_update.embedding
        if chunk_update.metadata is not None:
            chunk.metadata = chunk_update.metadata

        chunk.updated_at = datetime.utcnow()
        updated_chunk = await chunk_repo.update(chunk_id, chunk)

        # Update document timestamp
        document = await document_repo.get(chunk.document_id)
        document.updated_at = datetime.utcnow()
        await document_repo.update(document.id, document)

        # Update in vector index if embedding changed
        if chunk_update.embedding is not None:
            await library_repo.update_chunk_in_index(document.library_id, updated_chunk)

        return updated_chunk

    async def delete_chunk(self, chunk_id: UUID) -> None:
        """
        Delete a chunk.

        Args:
            chunk_id: The chunk ID

        Raises:
            NotFoundException: If chunk not found
        """
        chunk_repo = get_chunk_repository()
        document_repo = get_document_repository()
        library_repo = get_library_repository()

        chunk = await chunk_repo.get(chunk_id)

        # Update document timestamp
        document = await document_repo.get(chunk.document_id)
        document.updated_at = datetime.utcnow()
        await document_repo.update(document.id, document)

        # Remove from index
        try:
            await library_repo.remove_chunk_from_index(document.library_id, chunk_id)
        except Exception:
            # Index errors shouldn't fail chunk deletion
            pass

        # Delete the chunk
        await chunk_repo.delete(chunk_id)


# Singleton instance
chunk_service = ChunkService()
