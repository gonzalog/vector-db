"""Document model definition."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from vector_db.models.chunk import Chunk, ChunkCreate


class DocumentMetadata(BaseModel):
    """Metadata associated with a document."""

    title: str | None = Field(None, description="Title of the document")
    author: str | None = Field(None, description="Author of the document")
    source: str | None = Field(None, description="Source of the document")
    document_type: str | None = Field(None, description="Type of document (pdf, txt, etc)")
    language: str | None = Field(None, description="Language of the document")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    custom: dict[str, Any] = Field(
        default_factory=dict, description="Custom metadata fields"
    )


class Document(BaseModel):
    """A document represents a collection of chunks with metadata."""

    id: UUID = Field(
        default_factory=uuid4, description="Unique identifier for the document"
    )
    library_id: UUID = Field(..., description="ID of the parent library")
    name: str = Field(..., min_length=1, description="Name of the document")
    chunks: list[Chunk] = Field(
        default_factory=list, description="Chunks that make up the document"
    )
    metadata: DocumentMetadata = Field(
        default_factory=DocumentMetadata, description="Associated metadata"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "library_id": "123e4567-e89b-12d3-a456-426614174000",
                "name": "Sample Document",
                "metadata": {
                    "title": "A Sample Document",
                    "author": "John Doe",
                    "document_type": "pdf",
                    "tags": ["sample", "test"],
                },
            }
        }
    )


class DocumentCreate(BaseModel):
    """Schema for creating a new document."""

    name: str = Field(..., min_length=1, description="Name of the document")
    chunks: list[ChunkCreate] = Field(
        default_factory=list, description="Initial chunks for the document"
    )
    metadata: DocumentMetadata = Field(
        default_factory=DocumentMetadata, description="Associated metadata"
    )


class DocumentUpdate(BaseModel):
    """Schema for updating a document."""

    name: str | None = Field(None, min_length=1, description="Name of the document")
    metadata: DocumentMetadata | None = Field(None, description="Associated metadata")
