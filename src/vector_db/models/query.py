"""Query models for vector search."""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from vector_db.models.chunk import Chunk


class VectorQuery(BaseModel):
    """Schema for a vector similarity search query."""

    embedding: list[float] = Field(..., description="Query embedding vector")
    k: int = Field(default=10, ge=1, le=1000, description="Number of results to return")
    metadata_filter: dict[str, Any] | None = Field(
        None, description="Metadata filters to apply"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                "k": 5,
                "metadata_filter": {"tags": ["important"], "author": "John Doe"},
            }
        }
    )


class SearchResult(BaseModel):
    """A single search result with similarity score."""

    chunk: Chunk = Field(..., description="The matching chunk")
    score: float = Field(..., description="Similarity score (higher is more similar)")
    distance: float = Field(..., description="Distance metric value")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunk": {
                    "id": "123e4567-e89b-12d3-a456-426614174000",
                    "document_id": "123e4567-e89b-12d3-a456-426614174001",
                    "text": "Sample text content",
                    "embedding": [0.1, 0.2, 0.3],
                    "metadata": {"source": "doc.pdf"},
                },
                "score": 0.95,
                "distance": 0.05,
            }
        }
    )


class SearchResponse(BaseModel):
    """Response from a vector search query."""

    query: VectorQuery = Field(..., description="The original query")
    results: list[SearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results returned")
    library_id: UUID = Field(..., description="ID of the library searched")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": {"embedding": [0.1, 0.2, 0.3], "k": 5},
                "results": [],
                "total_results": 0,
                "library_id": "123e4567-e89b-12d3-a456-426614174000",
            }
        }
    )
