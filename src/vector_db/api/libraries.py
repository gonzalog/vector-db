"""API endpoints for library operations."""

from typing import Union
from uuid import UUID

from fastapi import APIRouter, Query, status

from vector_db.models import (
    Document,
    Library,
    LibraryCreate,
    LibraryUpdate,
    PaginatedResponse,
    SearchResponse,
    VectorQuery,
)
from vector_db.services.library_service import library_service

router = APIRouter(prefix="/libraries", tags=["libraries"])


@router.post("", response_model=Library, status_code=status.HTTP_201_CREATED)
async def create_library(library_create: LibraryCreate) -> Library:
    """Create a new library."""
    return await library_service.create_library(library_create)


@router.get(
    "",
    response_model=PaginatedResponse[Library],
    status_code=status.HTTP_200_OK,
)
def get_all_libraries(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum items to return"),
) -> PaginatedResponse[Library]:
    """Get all libraries with pagination."""
    return library_service.get_libraries_paginated(skip=skip, limit=limit)


@router.get("/{library_id}", response_model=Library, status_code=status.HTTP_200_OK)
def get_library(library_id: UUID) -> Library:
    """Get a library by ID."""
    return library_service.get_library(library_id)


@router.put("/{library_id}", response_model=Library, status_code=status.HTTP_200_OK)
async def update_library(library_id: UUID, library_update: LibraryUpdate) -> Library:
    """Update a library."""
    return await library_service.update_library(library_id, library_update)


@router.delete("/{library_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_library(library_id: UUID) -> None:
    """Delete a library."""
    await library_service.delete_library(library_id)


@router.get(
    "/{library_id}/documents",
    response_model=list[Document],
    status_code=status.HTTP_200_OK,
)
def get_library_documents(library_id: UUID) -> list[Document]:
    """Get all documents in a library."""
    return library_service.get_documents(library_id)


@router.post(
    "/{library_id}/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
)
def search_library(library_id: UUID, query: VectorQuery) -> SearchResponse:
    """Search for similar vectors in a library."""
    return library_service.search_library(library_id, query)
