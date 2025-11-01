"""Unit tests for the Vector DB SDK client."""

from unittest.mock import Mock, patch
from uuid import UUID, uuid4

import httpx
import pytest

from vector_db.sdk import (
    VectorDBClient,
    NotFoundError,
    AlreadyExistsError,
    ValidationError,
    ServerError,
    ConnectionError as SDKConnectionError,
)


@pytest.fixture
def client():
    """Create a VectorDBClient instance for testing."""
    return VectorDBClient(base_url="http://test-api.com", timeout=10.0)


@pytest.fixture
def library_id():
    """Generate a test library ID."""
    return uuid4()


@pytest.fixture
def document_id():
    """Generate a test document ID."""
    return uuid4()


@pytest.fixture
def chunk_id():
    """Generate a test chunk ID."""
    return uuid4()


class TestClientInitialization:
    """Tests for client initialization."""

    def test_client_init_with_defaults(self):
        """Test client initialization with default values."""
        client = VectorDBClient()
        assert client.base_url == "http://localhost:8000"
        assert client.api_prefix == "/api/v1"
        assert client.timeout == 30.0

    def test_client_init_with_custom_values(self):
        """Test client initialization with custom values."""
        client = VectorDBClient(
            base_url="http://custom.api.com",
            api_prefix="/v2",
            timeout=60.0,
        )
        assert client.base_url == "http://custom.api.com"
        assert client.api_prefix == "/v2"
        assert client.timeout == 60.0

    def test_client_context_manager(self):
        """Test client as context manager."""
        with patch.object(httpx.Client, "close") as mock_close:
            with VectorDBClient() as client:
                assert isinstance(client, VectorDBClient)
            mock_close.assert_called_once()


class TestLibraryOperations:
    """Tests for library-related operations."""

    def test_create_library(self, client, library_id):
        """Test creating a library."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": str(library_id),
            "name": "Test Library",
            "index_config": {
                "index_type": "flat",
                "distance_metric": "cosine",
                "n_hash_tables": None,
                "n_hash_bits": None,
                "M": None,
                "ef_construction": None,
                "ef_search": None,
            },
            "metadata": {},
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T00:00:00",
        }

        with patch.object(client._client, "post", return_value=mock_response):
            library = client.create_library(name="Test Library")

            assert library.id == library_id
            assert library.name == "Test Library"
            assert library.index_config.index_type == "flat"

    def test_get_library(self, client, library_id):
        """Test getting a library by ID."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": str(library_id),
            "name": "Test Library",
            "index_config": {
                "index_type": "flat",
                "distance_metric": "cosine",
                "n_hash_tables": None,
                "n_hash_bits": None,
                "M": None,
                "ef_construction": None,
                "ef_search": None,
            },
            "metadata": {},
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T00:00:00",
        }

        with patch.object(client._client, "get", return_value=mock_response):
            library = client.get_library(library_id)

            assert library.id == library_id
            assert library.name == "Test Library"

    def test_list_libraries(self, client, library_id):
        """Test listing libraries with pagination."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "items": [
                {
                    "id": str(library_id),
                    "name": "Test Library",
                    "index_config": {
                        "index_type": "flat",
                        "distance_metric": "cosine",
                        "n_hash_tables": None,
                        "n_hash_bits": None,
                        "M": None,
                        "ef_construction": None,
                        "ef_search": None,
                    },
                    "metadata": {},
                    "created_at": "2025-01-01T00:00:00",
                    "updated_at": "2025-01-01T00:00:00",
                }
            ],
            "total": 1,
            "skip": 0,
            "limit": 50,
            "has_more": False,
        }

        with patch.object(client._client, "get", return_value=mock_response):
            result = client.list_libraries(skip=0, limit=50)

            assert result.total == 1
            assert len(result.items) == 1
            assert result.items[0].id == library_id

    def test_update_library(self, client, library_id):
        """Test updating a library."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": str(library_id),
            "name": "Updated Library",
            "index_config": {
                "index_type": "flat",
                "distance_metric": "cosine",
                "n_hash_tables": None,
                "n_hash_bits": None,
                "M": None,
                "ef_construction": None,
                "ef_search": None,
            },
            "metadata": {},
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T00:00:00",
        }

        with patch.object(client._client, "put", return_value=mock_response):
            library = client.update_library(library_id, name="Updated Library")

            assert library.name == "Updated Library"

    def test_delete_library(self, client, library_id):
        """Test deleting a library."""
        mock_response = Mock()
        mock_response.status_code = 204

        with patch.object(client._client, "delete", return_value=mock_response):
            result = client.delete_library(library_id)

            assert result is None

    def test_get_library_documents(self, client, library_id, document_id):
        """Test getting all documents in a library."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "id": str(document_id),
                "library_id": str(library_id),
                "name": "Test Document",
                "metadata": {},
                "created_at": "2025-01-01T00:00:00",
                "updated_at": "2025-01-01T00:00:00",
            }
        ]

        with patch.object(client._client, "get", return_value=mock_response):
            documents = client.get_library_documents(library_id)

            assert len(documents) == 1
            assert documents[0].id == document_id

    def test_search_library(self, client, library_id, chunk_id):
        """Test searching a library."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "query": {
                "embedding": [0.1, 0.2, 0.3],
                "k": 10,
                "metadata_filter": None,
            },
            "results": [
                {
                    "chunk": {
                        "id": str(chunk_id),
                        "document_id": str(uuid4()),
                        "text": "Test chunk",
                        "embedding": [0.1, 0.2, 0.3],
                        "metadata": {
                            "source": None,
                            "page_number": None,
                            "position": None,
                            "author": None,
                            "tags": [],
                            "custom": {},
                        },
                        "created_at": "2025-01-01T00:00:00",
                        "updated_at": "2025-01-01T00:00:00",
                    },
                    "score": 0.95,
                    "distance": 0.05,
                    "rank": 0,
                }
            ],
            "total_results": 1,
            "library_id": str(library_id),
        }

        with patch.object(client._client, "post", return_value=mock_response):
            results = client.search_library(
                library_id, query=[0.1, 0.2, 0.3], top_k=10
            )

            assert results.total_results == 1
            assert len(results.results) == 1
            assert results.results[0].score == 0.95


class TestDocumentOperations:
    """Tests for document-related operations."""

    def test_create_document(self, client, library_id, document_id):
        """Test creating a document."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": str(document_id),
            "library_id": str(library_id),
            "name": "Test Document",
            "metadata": {},
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T00:00:00",
        }

        with patch.object(client._client, "post", return_value=mock_response):
            document = client.create_document(library_id, name="Test Document")

            assert document.id == document_id
            assert document.library_id == library_id
            assert document.name == "Test Document"

    def test_get_document(self, client, document_id, library_id):
        """Test getting a document by ID."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": str(document_id),
            "library_id": str(library_id),
            "name": "Test Document",
            "metadata": {},
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T00:00:00",
        }

        with patch.object(client._client, "get", return_value=mock_response):
            document = client.get_document(document_id)

            assert document.id == document_id
            assert document.name == "Test Document"

    def test_list_documents(self, client, library_id, document_id):
        """Test listing documents in a library."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "items": [
                {
                    "id": str(document_id),
                    "library_id": str(library_id),
                    "name": "Test Document",
                    "metadata": {},
                    "created_at": "2025-01-01T00:00:00",
                    "updated_at": "2025-01-01T00:00:00",
                }
            ],
            "total": 1,
            "skip": 0,
            "limit": 50,
            "has_more": False,
        }

        with patch.object(client._client, "get", return_value=mock_response):
            result = client.list_documents(library_id, skip=0, limit=50)

            assert result.total == 1
            assert len(result.items) == 1
            assert result.items[0].id == document_id

    def test_update_document(self, client, document_id, library_id):
        """Test updating a document."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": str(document_id),
            "library_id": str(library_id),
            "name": "Updated Document",
            "metadata": {},
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T00:00:00",
        }

        with patch.object(client._client, "put", return_value=mock_response):
            document = client.update_document(document_id, name="Updated Document")

            assert document.name == "Updated Document"

    def test_delete_document(self, client, document_id):
        """Test deleting a document."""
        mock_response = Mock()
        mock_response.status_code = 204

        with patch.object(client._client, "delete", return_value=mock_response):
            result = client.delete_document(document_id)

            assert result is None

    def test_get_document_chunks(self, client, document_id, chunk_id):
        """Test getting all chunks in a document."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "id": str(chunk_id),
                "document_id": str(document_id),
                "text": "Test chunk",
                "embedding": [0.1, 0.2, 0.3],
                "metadata": {
                    "source": None,
                    "page_number": None,
                    "position": None,
                    "author": None,
                    "tags": [],
                    "custom": {},
                },
                "created_at": "2025-01-01T00:00:00",
                "updated_at": "2025-01-01T00:00:00",
            }
        ]

        with patch.object(client._client, "get", return_value=mock_response):
            chunks = client.get_document_chunks(document_id)

            assert len(chunks) == 1
            assert chunks[0].id == chunk_id


class TestChunkOperations:
    """Tests for chunk-related operations."""

    def test_create_chunk(self, client, document_id, chunk_id):
        """Test creating a chunk."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": str(chunk_id),
            "document_id": str(document_id),
            "text": "Test chunk",
            "embedding": [0.1, 0.2, 0.3],
            "metadata": {
                "source": None,
                "page_number": None,
                "position": None,
                "author": None,
                "tags": [],
                "custom": {},
            },
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T00:00:00",
        }

        with patch.object(client._client, "post", return_value=mock_response):
            chunk = client.create_chunk(
                document_id, text="Test chunk", embedding=[0.1, 0.2, 0.3]
            )

            assert chunk.id == chunk_id
            assert chunk.text == "Test chunk"

    def test_get_chunk(self, client, chunk_id, document_id):
        """Test getting a chunk by ID."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": str(chunk_id),
            "document_id": str(document_id),
            "text": "Test chunk",
            "embedding": [0.1, 0.2, 0.3],
            "metadata": {
                "source": None,
                "page_number": None,
                "position": None,
                "author": None,
                "tags": [],
                "custom": {},
            },
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T00:00:00",
        }

        with patch.object(client._client, "get", return_value=mock_response):
            chunk = client.get_chunk(chunk_id)

            assert chunk.id == chunk_id
            assert chunk.text == "Test chunk"

    def test_list_chunks(self, client, document_id, chunk_id):
        """Test listing chunks in a document."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "id": str(chunk_id),
                "document_id": str(document_id),
                "text": "Test chunk",
                "embedding": [0.1, 0.2, 0.3],
                "metadata": {
                    "source": None,
                    "page_number": None,
                    "position": None,
                    "author": None,
                    "tags": [],
                    "custom": {},
                },
                "created_at": "2025-01-01T00:00:00",
                "updated_at": "2025-01-01T00:00:00",
            }
        ]

        with patch.object(client._client, "get", return_value=mock_response):
            chunks = client.list_chunks(document_id)

            assert len(chunks) == 1
            assert chunks[0].id == chunk_id

    def test_update_chunk(self, client, chunk_id, document_id):
        """Test updating a chunk."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": str(chunk_id),
            "document_id": str(document_id),
            "text": "Updated chunk",
            "embedding": [0.1, 0.2, 0.3],
            "metadata": {
                "source": None,
                "page_number": None,
                "position": None,
                "author": None,
                "tags": [],
                "custom": {},
            },
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T00:00:00",
        }

        with patch.object(client._client, "put", return_value=mock_response):
            chunk = client.update_chunk(chunk_id, text="Updated chunk")

            assert chunk.text == "Updated chunk"

    def test_delete_chunk(self, client, chunk_id):
        """Test deleting a chunk."""
        mock_response = Mock()
        mock_response.status_code = 204

        with patch.object(client._client, "delete", return_value=mock_response):
            result = client.delete_chunk(chunk_id)

            assert result is None


class TestErrorHandling:
    """Tests for error handling in the SDK."""

    def test_not_found_error(self, client, library_id):
        """Test handling of 404 errors."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_response.json.return_value = {"message": "Library not found"}

        with patch.object(client._client, "get") as mock_get:
            mock_get.return_value = mock_response
            mock_get.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
                "404",
                request=Mock(),
                response=mock_response,
            )

            with pytest.raises(NotFoundError) as exc_info:
                client.get_library(library_id)

            assert "Library not found" in str(exc_info.value)

    def test_already_exists_error(self, client):
        """Test handling of 409 errors."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 409
        mock_response.json.return_value = {"message": "Library already exists"}

        with patch.object(client._client, "post") as mock_post:
            mock_post.return_value = mock_response
            mock_post.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
                "409",
                request=Mock(),
                response=mock_response,
            )

            with pytest.raises(AlreadyExistsError) as exc_info:
                client.create_library(name="Test Library")

            assert "already exists" in str(exc_info.value).lower()

    def test_validation_error(self, client, document_id):
        """Test handling of 422 validation errors."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 422
        mock_response.json.return_value = {"message": "Validation error"}

        with patch.object(client, "_make_request") as mock_request:
            mock_request.return_value = mock_response
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "422",
                request=Mock(),
                response=mock_response,
            )

            with pytest.raises(ValidationError) as exc_info:
                # Use valid data that passes client-side validation
                # The mocked API will return validation error
                client.create_chunk(document_id, text="valid text", embedding=[0.1, 0.2])

            assert "Validation error" in str(exc_info.value)

    def test_server_error(self, client, library_id):
        """Test handling of 500 server errors."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 500
        mock_response.json.return_value = {"message": "Internal server error"}

        with patch.object(client._client, "get") as mock_get:
            mock_get.return_value = mock_response
            mock_get.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
                "500",
                request=Mock(),
                response=mock_response,
            )

            with pytest.raises(ServerError) as exc_info:
                client.get_library(library_id)

            assert exc_info.value.status_code == 500

    def test_connection_error(self, client, library_id):
        """Test handling of connection errors."""
        # Patch the underlying httpx client to raise ConnectError
        # The _make_request method will catch it and wrap it in SDKConnectionError
        with patch.object(client._client, "get", side_effect=httpx.ConnectError("Connection failed")):
            with pytest.raises(SDKConnectionError) as exc_info:
                client.get_library(library_id)

            assert "Failed to connect" in str(exc_info.value)
