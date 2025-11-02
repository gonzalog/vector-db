"""Library model definition."""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


# Valid index types and distance metrics
VALID_INDEX_TYPES = {"flat", "lsh", "hnsw"}
VALID_DISTANCE_METRICS = {"cosine", "euclidean", "dot_product"}


class IndexConfig(BaseModel):
    """Configuration for vector index."""

    index_type: Literal["flat", "lsh", "hnsw"] = Field(
        default="flat", description="Type of index (flat, lsh, hnsw)"
    )
    distance_metric: Literal["cosine", "euclidean", "dot_product"] = Field(
        default="cosine", description="Distance metric (cosine, euclidean, dot_product)"
    )
    # LSH specific parameters
    n_hash_tables: int | None = Field(
        None, description="Number of hash tables for LSH index (default: 5)", gt=0
    )
    n_hash_bits: int | None = Field(
        None, description="Number of hash bits for LSH index (default: 8)", gt=0
    )
    # HNSW specific parameters
    M: int | None = Field(
        None, description="Number of bi-directional links for HNSW index (default: 16)", gt=0
    )
    ef_construction: int | None = Field(
        None,
        description="Size of dynamic candidate list during construction for HNSW (default: 200)",
        gt=0,
    )
    ef_search: int | None = Field(
        None,
        description="Size of dynamic candidate list during search for HNSW (default: 50)",
        gt=0,
    )

    @field_validator("n_hash_tables", "n_hash_bits")
    @classmethod
    def validate_lsh_params(cls, v: int | None, info) -> int | None:
        """Validate LSH-specific parameters."""
        if v is not None and info.data.get("index_type") != "lsh":
            raise ValueError(
                f"{info.field_name} can only be set for LSH index type"
            )
        return v

    @field_validator("M", "ef_construction", "ef_search")
    @classmethod
    def validate_hnsw_params(cls, v: int | None, info) -> int | None:
        """Validate HNSW-specific parameters."""
        if v is not None and info.data.get("index_type") != "hnsw":
            raise ValueError(
                f"{info.field_name} can only be set for HNSW index type"
            )
        return v


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
    metadata: LibraryMetadata = Field(
        default_factory=LibraryMetadata, description="Associated metadata"
    )
    index_config: IndexConfig = Field(
        default_factory=IndexConfig, description="Vector index configuration"
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
    metadata: LibraryMetadata = Field(
        default_factory=LibraryMetadata, description="Associated metadata"
    )
    index_config: IndexConfig = Field(
        default_factory=IndexConfig, description="Vector index configuration"
    )


class LibraryUpdate(BaseModel):
    """Schema for updating a library."""

    name: str | None = Field(None, min_length=1, description="Name of the library")
    metadata: LibraryMetadata | None = Field(None, description="Associated metadata")
