"""Service for chunk operations."""

from datetime import datetime
from uuid import UUID

from vector_db.core.exceptions import NotFoundException
from vector_db.models import Chunk, ChunkCreate, ChunkUpdate
from vector_db.repositories.memory_repository import (
    chunk_repository,
    document_repository,
)


class ChunkService:
    """Service for managing chunks."""

    def create_chunk(self, document_id: UUID, chunk_create: ChunkCreate) -> Chunk:
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
        # Verify document exists
        document = document_repository.get(document_id)

        # Create the chunk
        chunk = Chunk(
            document_id=document_id,
            text=chunk_create.text,
            embedding=chunk_create.embedding,
            metadata=chunk_create.metadata,
        )

        # Save chunk
        chunk = chunk_repository.create(chunk)

        # Add chunk to document
        document.chunks.append(chunk)
        document.updated_at = datetime.utcnow()
        document_repository.update(document_id, document)

        return chunk

    def get_chunk(self, chunk_id: UUID) -> Chunk:
        """
        Get a chunk by ID.

        Args:
            chunk_id: The chunk ID

        Returns:
            The chunk

        Raises:
            NotFoundException: If chunk not found
        """
        return chunk_repository.get(chunk_id)

    def get_all_chunks(self, document_id: UUID) -> list[Chunk]:
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

    def update_chunk(self, chunk_id: UUID, chunk_update: ChunkUpdate) -> Chunk:
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
        chunk = chunk_repository.get(chunk_id)

        if chunk_update.text is not None:
            chunk.text = chunk_update.text
        if chunk_update.embedding is not None:
            chunk.embedding = chunk_update.embedding
        if chunk_update.metadata is not None:
            chunk.metadata = chunk_update.metadata

        chunk.updated_at = datetime.utcnow()
        return chunk_repository.update(chunk_id, chunk)

    def delete_chunk(self, chunk_id: UUID) -> None:
        """
        Delete a chunk.

        Args:
            chunk_id: The chunk ID

        Raises:
            NotFoundException: If chunk not found
        """
        chunk = chunk_repository.get(chunk_id)

        # Remove chunk from document
        document = document_repository.get(chunk.document_id)
        document.chunks = [c for c in document.chunks if c.id != chunk_id]
        document.updated_at = datetime.utcnow()
        document_repository.update(document.id, document)

        # Delete the chunk
        chunk_repository.delete(chunk_id)


# Singleton instance
chunk_service = ChunkService()
