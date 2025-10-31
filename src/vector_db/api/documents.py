"""API endpoints for document operations."""

from typing import Union
from uuid import UUID

from fastapi import APIRouter, Query, status

from vector_db.models import (
    Chunk,
    Document,
    DocumentCreate,
    DocumentUpdate,
    PaginatedResponse,
)
from vector_db.services.document_service import document_service

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=Document, status_code=status.HTTP_201_CREATED)
async def create_document(library_id: UUID, document_create: DocumentCreate) -> Document:
    """Create a new document in a library."""
    return await document_service.create_document(library_id, document_create)


@router.get("/{document_id}", response_model=Document, status_code=status.HTTP_200_OK)
def get_document(document_id: UUID) -> Document:
    """Get a document by ID."""
    return document_service.get_document(document_id)


@router.get(
    "",
    response_model=PaginatedResponse[Document],
    status_code=status.HTTP_200_OK,
)
def get_all_documents(
    library_id: UUID,
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum items to return"),
) -> PaginatedResponse[Document]:
    """Get all documents in a library with pagination."""
    return document_service.get_documents_paginated(
        library_id=library_id, skip=skip, limit=limit
    )


@router.put("/{document_id}", response_model=Document, status_code=status.HTTP_200_OK)
def update_document(document_id: UUID, document_update: DocumentUpdate) -> Document:
    """Update a document."""
    return document_service.update_document(document_id, document_update)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: UUID) -> None:
    """Delete a document."""
    await document_service.delete_document(document_id)


@router.get(
    "/{document_id}/chunks",
    response_model=list[Chunk],
    status_code=status.HTTP_200_OK,
)
def get_document_chunks(document_id: UUID) -> list[Chunk]:
    """Get all chunks in a document."""
    return document_service.get_chunks(document_id)
