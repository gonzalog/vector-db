"""Vector DB Python SDK Client."""

from typing import Any
from uuid import UUID

import httpx

from vector_db.models import (
    Chunk,
    ChunkCreate,
    ChunkUpdate,
    Document,
    DocumentCreate,
    DocumentUpdate,
    Library,
    LibraryCreate,
    LibraryUpdate,
    PaginatedResponse,
    SearchResponse,
    VectorQuery,
)
from vector_db.sdk.exceptions import (
    AlreadyExistsError,
    ConnectionError,
    NotFoundError,
    ServerError,
    ValidationError,
    VectorDBSDKError,
)


class VectorDBClient:
    """Python SDK client for Vector DB API.

    Example:
        ```python
        from vector_db.sdk import VectorDBClient

        # Initialize client
        client = VectorDBClient(base_url="http://localhost:8000")

        # Create a library
        library = client.create_library(name="My Library")

        # Search in library
        results = client.search_library(
            library_id=library.id,
            query=[0.1, 0.2, 0.3, ...],
            top_k=10
        )
        ```
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_prefix: str = "/api/v1",
        timeout: float = 30.0,
    ):
        """Initialize the Vector DB client.

        Args:
            base_url: Base URL of the Vector DB API
            api_prefix: API prefix (default: /api/v1)
            timeout: Request timeout in seconds (default: 30.0)
        """
        self.base_url = base_url.rstrip("/")
        self.api_prefix = api_prefix
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def _make_url(self, path: str) -> str:
        """Construct full URL from path."""
        return f"{self.base_url}{self.api_prefix}{path}"

    def _make_request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make an HTTP request and handle connection errors."""
        try:
            if method == "GET":
                return self._client.get(url, **kwargs)
            elif method == "POST":
                return self._client.post(url, **kwargs)
            elif method == "PUT":
                return self._client.put(url, **kwargs)
            elif method == "DELETE":
                return self._client.delete(url, **kwargs)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
        except httpx.RequestError as e:
            raise ConnectionError(f"Failed to connect to API: {str(e)}") from e

    def _handle_response(self, response: httpx.Response) -> Any:
        """Handle HTTP response and raise appropriate exceptions."""
        try:
            response.raise_for_status()
            if response.status_code == 204:
                return None
            return response.json()
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            try:
                error_data = e.response.json()
                message = error_data.get("message", str(e))
            except Exception:
                message = str(e)

            if status_code == 404:
                raise NotFoundError(message) from e
            elif status_code == 409:
                raise AlreadyExistsError(message) from e
            elif status_code == 422:
                raise ValidationError(message) from e
            elif status_code >= 500:
                raise ServerError(message, status_code) from e
            else:
                raise VectorDBSDKError(message, status_code) from e

    # Library Operations

    def create_library(
        self,
        name: str,
        index_type: str = "flat",
        distance_metric: str = "cosine",
        metadata: dict | None = None,
        index_config: dict | None = None,
        **index_params,
    ) -> Library:
        """Create a new library.

        Args:
            name: Name of the library
            index_type: Type of vector index (flat, lsh, hnsw)
            distance_metric: Distance metric (cosine, euclidean, dot_product)
            metadata: Optional library metadata
            index_config: Optional complete index config dict
            **index_params: Index-specific parameters (M, ef_construction, n_hash_tables, etc.)

        Returns:
            Created library object
        """
        from vector_db.models import IndexConfig

        # Build index config
        if index_config is None:
            index_config_obj = IndexConfig(
                index_type=index_type,
                distance_metric=distance_metric,
                **index_params
            )
        else:
            # Use provided index_config dict
            index_config_obj = IndexConfig(**index_config)

        # Build library create request
        library_create = LibraryCreate(
            name=name,
            index_config=index_config_obj,
            metadata=metadata or {}
        )

        data = library_create.model_dump(mode="json")
        response = self._make_request("POST", self._make_url("/libraries"), json=data)
        return Library(**self._handle_response(response))

    def get_library(self, library_id: UUID | str) -> Library:
        """Get a library by ID.

        Args:
            library_id: UUID of the library

        Returns:
            Library object
        """
        response = self._make_request("GET", self._make_url(f"/libraries/{library_id}"))
        return Library(**self._handle_response(response))

    def list_libraries(
        self, skip: int = 0, limit: int = 50
    ) -> PaginatedResponse[Library]:
        """List all libraries with pagination.

        Args:
            skip: Number of items to skip
            limit: Maximum number of items to return

        Returns:
            Paginated response with libraries
        """
        response = self._make_request(
            "GET", self._make_url("/libraries"), params={"skip": skip, "limit": limit}
        )
        data = self._handle_response(response)
        return PaginatedResponse[Library](
            items=[Library(**item) for item in data["items"]],
            total=data["total"],
            skip=data["skip"],
            limit=data["limit"],
            has_more=data["has_more"],
        )

    def update_library(
        self, library_id: UUID | str, name: str | None = None, **kwargs
    ) -> Library:
        """Update a library.

        Args:
            library_id: UUID of the library
            name: New name for the library
            **kwargs: Additional library update parameters

        Returns:
            Updated library object
        """
        data = LibraryUpdate(name=name, **kwargs).model_dump(
            mode="json", exclude_unset=True
        )

        response = self._make_request("PUT", self._make_url(f"/libraries/{library_id}"), json=data)
        return Library(**self._handle_response(response))

    def delete_library(self, library_id: UUID | str) -> None:
        """Delete a library.

        Args:
            library_id: UUID of the library
        """
        response = self._make_request("DELETE", self._make_url(f"/libraries/{library_id}"))
        self._handle_response(response)

    def get_library_documents(self, library_id: UUID | str) -> list[Document]:
        """Get all documents in a library.

        Args:
            library_id: UUID of the library

        Returns:
            List of documents
        """
        response = self._make_request(
            "GET", self._make_url(f"/libraries/{library_id}/documents")
        )
        data = self._handle_response(response)
        return [Document(**item) for item in data]

    def search_library(
        self,
        library_id: UUID | str,
        query: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> SearchResponse:
        """Search for similar vectors in a library.

        Args:
            library_id: UUID of the library
            query: Query vector (list of floats)
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            Search response with results
        """
        query_data = VectorQuery(
            embedding=query, k=top_k, metadata_filter=filters
        ).model_dump(mode="json")

        response = self._make_request(
            "POST", self._make_url(f"/libraries/{library_id}/search"), json=query_data
        )
        return SearchResponse(**self._handle_response(response))

    # Document Operations

    def create_document(
        self, library_id: UUID | str, name: str, **kwargs
    ) -> Document:
        """Create a new document in a library.

        Args:
            library_id: UUID of the library
            name: Name of the document
            **kwargs: Additional document creation parameters

        Returns:
            Created document object
        """
        data = DocumentCreate(name=name, **kwargs).model_dump(mode="json")

        response = self._make_request(
            "POST", self._make_url("/documents"), params={"library_id": str(library_id)}, json=data
        )
        return Document(**self._handle_response(response))

    def get_document(self, document_id: UUID | str) -> Document:
        """Get a document by ID.

        Args:
            document_id: UUID of the document

        Returns:
            Document object
        """
        response = self._make_request("GET", self._make_url(f"/documents/{document_id}"))
        return Document(**self._handle_response(response))

    def list_documents(
        self, library_id: UUID | str, skip: int = 0, limit: int = 50
    ) -> PaginatedResponse[Document]:
        """List all documents in a library with pagination.

        Args:
            library_id: UUID of the library
            skip: Number of items to skip
            limit: Maximum number of items to return

        Returns:
            Paginated response with documents
        """
        response = self._make_request(
            "GET", self._make_url("/documents"),
            params={"library_id": str(library_id), "skip": skip, "limit": limit},
        )
        data = self._handle_response(response)
        return PaginatedResponse[Document](
            items=[Document(**item) for item in data["items"]],
            total=data["total"],
            skip=data["skip"],
            limit=data["limit"],
            has_more=data["has_more"],
        )

    def update_document(
        self, document_id: UUID | str, name: str | None = None, **kwargs
    ) -> Document:
        """Update a document.

        Args:
            document_id: UUID of the document
            name: New name for the document
            **kwargs: Additional document update parameters

        Returns:
            Updated document object
        """
        data = DocumentUpdate(name=name, **kwargs).model_dump(
            mode="json", exclude_unset=True
        )

        response = self._make_request(
            "PUT", self._make_url(f"/documents/{document_id}"), json=data
        )
        return Document(**self._handle_response(response))

    def delete_document(self, document_id: UUID | str) -> None:
        """Delete a document.

        Args:
            document_id: UUID of the document
        """
        response = self._make_request("DELETE", self._make_url(f"/documents/{document_id}"))
        self._handle_response(response)

    def get_document_chunks(self, document_id: UUID | str) -> list[Chunk]:
        """Get all chunks in a document.

        Args:
            document_id: UUID of the document

        Returns:
            List of chunks
        """
        response = self._make_request("GET", self._make_url(f"/documents/{document_id}/chunks"))
        data = self._handle_response(response)
        return [Chunk(**item) for item in data]

    # Chunk Operations

    def create_chunk(
        self, document_id: UUID | str, text: str, embedding: list[float], **kwargs
    ) -> Chunk:
        """Create a new chunk in a document.

        Args:
            document_id: UUID of the document
            text: Text content of the chunk
            embedding: Vector embedding of the chunk
            **kwargs: Additional chunk creation parameters

        Returns:
            Created chunk object
        """
        data = ChunkCreate(text=text, embedding=embedding, **kwargs).model_dump(
            mode="json"
        )

        response = self._make_request(
            "POST", self._make_url("/chunks"), params={"document_id": str(document_id)}, json=data
        )
        return Chunk(**self._handle_response(response))

    def get_chunk(self, chunk_id: UUID | str) -> Chunk:
        """Get a chunk by ID.

        Args:
            chunk_id: UUID of the chunk

        Returns:
            Chunk object
        """
        response = self._make_request("GET", self._make_url(f"/chunks/{chunk_id}"))
        return Chunk(**self._handle_response(response))

    def list_chunks(self, document_id: UUID | str) -> list[Chunk]:
        """List all chunks in a document.

        Args:
            document_id: UUID of the document

        Returns:
            List of chunks
        """
        response = self._make_request(
            "GET", self._make_url("/chunks"), params={"document_id": str(document_id)}
        )
        data = self._handle_response(response)
        return [Chunk(**item) for item in data]

    def update_chunk(
        self, chunk_id: UUID | str, text: str | None = None, **kwargs
    ) -> Chunk:
        """Update a chunk.

        Args:
            chunk_id: UUID of the chunk
            text: New text content
            **kwargs: Additional chunk update parameters

        Returns:
            Updated chunk object
        """
        data = ChunkUpdate(text=text, **kwargs).model_dump(
            mode="json", exclude_unset=True
        )

        response = self._make_request("PUT", self._make_url(f"/chunks/{chunk_id}"), json=data)
        return Chunk(**self._handle_response(response))

    def delete_chunk(self, chunk_id: UUID | str) -> None:
        """Delete a chunk.

        Args:
            chunk_id: UUID of the chunk
        """
        response = self._make_request("DELETE", self._make_url(f"/chunks/{chunk_id}"))
        self._handle_response(response)
