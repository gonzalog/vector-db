"""Chunk model definition."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ChunkMetadata(BaseModel):
    """Metadata associated with a chunk."""

    source: str | None = Field(None, description="Source of the chunk")
    page_number: int | None = Field(None, description="Page number in the document")
    position: int | None = Field(None, description="Position within the document")
    author: str | None = Field(None, description="Author of the content")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    custom: dict[str, Any] = Field(
        default_factory=dict, description="Custom metadata fields"
    )


class Chunk(BaseModel):
    """A chunk represents a piece of text with its embedding and metadata."""

    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the chunk")
    document_id: UUID = Field(..., description="ID of the parent document")
    text: str = Field(..., min_length=1, description="Text content of the chunk")
    embedding: list[float] = Field(..., description="Vector embedding of the text")
    metadata: ChunkMetadata = Field(
        default_factory=ChunkMetadata, description="Associated metadata"
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
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "text": "This is a sample chunk of text.",
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                "metadata": {
                    "source": "document.pdf",
                    "page_number": 1,
                    "position": 0,
                    "tags": ["sample", "test"],
                },
            }
        }
    )


class ChunkCreate(BaseModel):
    """Schema for creating a new chunk."""

    text: str = Field(..., min_length=1, description="Text content of the chunk")
    embedding: list[float] = Field(..., description="Vector embedding of the text")
    metadata: ChunkMetadata = Field(
        default_factory=ChunkMetadata, description="Associated metadata"
    )


class ChunkUpdate(BaseModel):
    """Schema for updating a chunk."""

    text: str | None = Field(None, min_length=1, description="Text content of the chunk")
    embedding: list[float] | None = Field(None, description="Vector embedding of the text")
    metadata: ChunkMetadata | None = Field(None, description="Associated metadata")
