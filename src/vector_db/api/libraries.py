"""API endpoints for library operations."""

from uuid import UUID

from fastapi import APIRouter, status

from vector_db.models import Document, Library, LibraryCreate, LibraryUpdate
from vector_db.services.library_service import library_service

router = APIRouter(prefix="/libraries", tags=["libraries"])


@router.post("", response_model=Library, status_code=status.HTTP_201_CREATED)
def create_library(library_create: LibraryCreate) -> Library:
    """Create a new library."""
    return library_service.create_library(library_create)


@router.get("", response_model=list[Library], status_code=status.HTTP_200_OK)
def get_all_libraries() -> list[Library]:
    """Get all libraries."""
    return library_service.get_all_libraries()


@router.get("/{library_id}", response_model=Library, status_code=status.HTTP_200_OK)
def get_library(library_id: UUID) -> Library:
    """Get a library by ID."""
    return library_service.get_library(library_id)


@router.put("/{library_id}", response_model=Library, status_code=status.HTTP_200_OK)
def update_library(library_id: UUID, library_update: LibraryUpdate) -> Library:
    """Update a library."""
    return library_service.update_library(library_id, library_update)


@router.delete("/{library_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_library(library_id: UUID) -> None:
    """Delete a library."""
    library_service.delete_library(library_id)


@router.get(
    "/{library_id}/documents",
    response_model=list[Document],
    status_code=status.HTTP_200_OK,
)
def get_library_documents(library_id: UUID) -> list[Document]:
    """Get all documents in a library."""
    return library_service.get_documents(library_id)
