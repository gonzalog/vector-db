"""API endpoints for chunk operations."""

from uuid import UUID

from fastapi import APIRouter, status

from vector_db.models import Chunk, ChunkCreate, ChunkUpdate
from vector_db.services.chunk_service import chunk_service

router = APIRouter(prefix="/chunks", tags=["chunks"])


@router.post("", response_model=Chunk, status_code=status.HTTP_201_CREATED)
def create_chunk(document_id: UUID, chunk_create: ChunkCreate) -> Chunk:
    """Create a new chunk in a document."""
    return chunk_service.create_chunk(document_id, chunk_create)


@router.get("/{chunk_id}", response_model=Chunk, status_code=status.HTTP_200_OK)
def get_chunk(chunk_id: UUID) -> Chunk:
    """Get a chunk by ID."""
    return chunk_service.get_chunk(chunk_id)


@router.get("", response_model=list[Chunk], status_code=status.HTTP_200_OK)
def get_all_chunks(document_id: UUID) -> list[Chunk]:
    """Get all chunks in a document."""
    return chunk_service.get_all_chunks(document_id)


@router.put("/{chunk_id}", response_model=Chunk, status_code=status.HTTP_200_OK)
def update_chunk(chunk_id: UUID, chunk_update: ChunkUpdate) -> Chunk:
    """Update a chunk."""
    return chunk_service.update_chunk(chunk_id, chunk_update)


@router.delete("/{chunk_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chunk(chunk_id: UUID) -> None:
    """Delete a chunk."""
    chunk_service.delete_chunk(chunk_id)
