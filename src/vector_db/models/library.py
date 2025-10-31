"""Library model definition."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from vector_db.models.document import Document, DocumentCreate


class LibraryMetadata(BaseModel):
    """Metadata associated with a library."""

    description: str | None = Field(None, description="Description of the library")
    owner: str | None = Field(None, description="Owner of the library")
    category: str | None = Field(None, description="Category of the library")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    embedding_model: str | None = Field(
        None, description="Name of the embedding model used"
    )
    embedding_dimension: int | None = Field(
        None, description="Dimension of the embeddings"
    )
    custom: dict[str, Any] = Field(
        default_factory=dict, description="Custom metadata fields"
    )


class Library(BaseModel):
    """A library represents a collection of documents."""

    id: UUID = Field(
        default_factory=uuid4, description="Unique identifier for the library"
    )
    name: str = Field(..., min_length=1, description="Name of the library")
    documents: list[Document] = Field(
        default_factory=list, description="Documents in the library"
    )
    metadata: LibraryMetadata = Field(
        default_factory=LibraryMetadata, description="Associated metadata"
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
                "name": "My Vector Library",
                "metadata": {
                    "description": "A collection of technical documents",
                    "owner": "john.doe@example.com",
                    "category": "technical",
                    "tags": ["ml", "ai", "nlp"],
                    "embedding_model": "cohere-embed-english-v3.0",
                    "embedding_dimension": 1024,
                },
            }
        }
    )


class LibraryCreate(BaseModel):
    """Schema for creating a new library."""

    name: str = Field(..., min_length=1, description="Name of the library")
    documents: list[DocumentCreate] = Field(
        default_factory=list, description="Initial documents for the library"
    )
    metadata: LibraryMetadata = Field(
        default_factory=LibraryMetadata, description="Associated metadata"
    )


class LibraryUpdate(BaseModel):
    """Schema for updating a library."""

    name: str | None = Field(None, min_length=1, description="Name of the library")
    metadata: LibraryMetadata | None = Field(None, description="Associated metadata")
