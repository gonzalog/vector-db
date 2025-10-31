"""Tests to verify that all API endpoints include UUIDs in responses."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from uuid import UUID

from vector_db.main import app
from vector_db.repositories.memory_repository import (
    chunk_repository,
    document_repository,
    library_repository,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_repositories():
    """Clear all repositories before each test."""
    library_repository.clear()
    document_repository.clear()
    chunk_repository.clear()
    yield
    library_repository.clear()
    document_repository.clear()
    chunk_repository.clear()


class TestLibraryUUIDs:
    """Test that library endpoints include UUIDs."""

    def test_create_library_returns_uuid(self):
        """Test that creating a library returns its UUID."""
        response = client.post(
            "/api/v1/libraries",
            json={"name": "Test Library"},
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()

        # Verify UUID is present and valid
        assert "id" in data
        assert data["id"] is not None
        # Verify it's a valid UUID by parsing it
        uuid_obj = UUID(data["id"])
        assert str(uuid_obj) == data["id"]

    def test_get_library_returns_uuid(self):
        """Test that getting a library returns its UUID."""
        # Create a library
        create_response = client.post(
            "/api/v1/libraries",
            json={"name": "Test Library"},
        )
        library_id = create_response.json()["id"]

        # Get the library
        response = client.get(f"/api/v1/libraries/{library_id}")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify UUID is present and matches
        assert "id" in data
        assert data["id"] == library_id

    def test_get_all_libraries_includes_uuids(self):
        """Test that listing libraries includes UUIDs for all items."""
        # Create multiple libraries
        library_ids = []
        for i in range(3):
            response = client.post(
                "/api/v1/libraries",
                json={"name": f"Library {i}"},
            )
            library_ids.append(response.json()["id"])

        # Get all libraries
        response = client.get("/api/v1/libraries")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify all items have UUIDs
        assert "items" in data
        assert len(data["items"]) == 3

        for item in data["items"]:
            assert "id" in item
            assert item["id"] is not None
            # Verify it's a valid UUID
            uuid_obj = UUID(item["id"])
            assert str(uuid_obj) == item["id"]
            # Verify it's one of our created libraries
            assert item["id"] in library_ids

    def test_update_library_returns_uuid(self):
        """Test that updating a library returns its UUID."""
        # Create a library
        create_response = client.post(
            "/api/v1/libraries",
            json={"name": "Original Name"},
        )
        library_id = create_response.json()["id"]

        # Update the library
        response = client.put(
            f"/api/v1/libraries/{library_id}",
            json={"name": "Updated Name"},
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify UUID is present and unchanged
        assert "id" in data
        assert data["id"] == library_id


class TestDocumentUUIDs:
    """Test that document endpoints include UUIDs."""

    def test_create_document_returns_uuid(self):
        """Test that creating a document returns its UUID."""
        # Create a library first
        library_response = client.post(
            "/api/v1/libraries",
            json={"name": "Test Library"},
        )
        library_id = library_response.json()["id"]

        # Create a document
        response = client.post(
            f"/api/v1/documents?library_id={library_id}",
            json={"name": "Test Document"},
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()

        # Verify UUIDs are present and valid
        assert "id" in data
        assert data["id"] is not None
        assert "library_id" in data
        assert data["library_id"] == library_id

        # Verify both are valid UUIDs
        UUID(data["id"])
        UUID(data["library_id"])

    def test_get_document_returns_uuid(self):
        """Test that getting a document returns its UUID."""
        # Create library and document
        library_response = client.post(
            "/api/v1/libraries",
            json={"name": "Test Library"},
        )
        library_id = library_response.json()["id"]

        create_response = client.post(
            f"/api/v1/documents?library_id={library_id}",
            json={"name": "Test Document"},
        )
        document_id = create_response.json()["id"]

        # Get the document
        response = client.get(f"/api/v1/documents/{document_id}")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify UUIDs are present
        assert "id" in data
        assert data["id"] == document_id
        assert "library_id" in data
        assert data["library_id"] == library_id

    def test_get_all_documents_includes_uuids(self):
        """Test that listing documents includes UUIDs for all items."""
        # Create a library
        library_response = client.post(
            "/api/v1/libraries",
            json={"name": "Test Library"},
        )
        library_id = library_response.json()["id"]

        # Create multiple documents
        document_ids = []
        for i in range(3):
            response = client.post(
                f"/api/v1/documents?library_id={library_id}",
                json={"name": f"Document {i}"},
            )
            document_ids.append(response.json()["id"])

        # Get all documents
        response = client.get(f"/api/v1/documents?library_id={library_id}")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify all items have UUIDs
        assert "items" in data
        assert len(data["items"]) == 3

        for item in data["items"]:
            assert "id" in item
            assert item["id"] is not None
            assert "library_id" in item
            assert item["library_id"] == library_id
            # Verify valid UUIDs
            UUID(item["id"])
            UUID(item["library_id"])
            # Verify it's one of our created documents
            assert item["id"] in document_ids

    def test_update_document_returns_uuid(self):
        """Test that updating a document returns its UUID."""
        # Create library and document
        library_response = client.post(
            "/api/v1/libraries",
            json={"name": "Test Library"},
        )
        library_id = library_response.json()["id"]

        create_response = client.post(
            f"/api/v1/documents?library_id={library_id}",
            json={"name": "Original Name"},
        )
        document_id = create_response.json()["id"]

        # Update the document
        response = client.put(
            f"/api/v1/documents/{document_id}",
            json={"name": "Updated Name"},
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify UUIDs are present and unchanged
        assert "id" in data
        assert data["id"] == document_id
        assert "library_id" in data
        assert data["library_id"] == library_id


class TestChunkUUIDs:
    """Test that chunk endpoints include UUIDs."""

    def test_create_chunk_returns_uuid(self):
        """Test that creating a chunk returns its UUID."""
        # Create library and document
        library_response = client.post(
            "/api/v1/libraries",
            json={"name": "Test Library"},
        )
        library_id = library_response.json()["id"]

        document_response = client.post(
            f"/api/v1/documents?library_id={library_id}",
            json={"name": "Test Document"},
        )
        document_id = document_response.json()["id"]

        # Create a chunk
        response = client.post(
            f"/api/v1/chunks?document_id={document_id}",
            json={
                "text": "Test chunk text",
                "embedding": [1.0, 2.0, 3.0],
            },
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()

        # Verify UUIDs are present and valid
        assert "id" in data
        assert data["id"] is not None
        assert "document_id" in data
        assert data["document_id"] == document_id

        # Verify both are valid UUIDs
        UUID(data["id"])
        UUID(data["document_id"])

    def test_get_chunk_returns_uuid(self):
        """Test that getting a chunk returns its UUID."""
        # Create library, document, and chunk
        library_response = client.post(
            "/api/v1/libraries",
            json={"name": "Test Library"},
        )
        library_id = library_response.json()["id"]

        document_response = client.post(
            f"/api/v1/documents?library_id={library_id}",
            json={"name": "Test Document"},
        )
        document_id = document_response.json()["id"]

        create_response = client.post(
            f"/api/v1/chunks?document_id={document_id}",
            json={
                "text": "Test chunk",
                "embedding": [1.0, 2.0, 3.0],
            },
        )
        chunk_id = create_response.json()["id"]

        # Get the chunk
        response = client.get(f"/api/v1/chunks/{chunk_id}")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify UUIDs are present
        assert "id" in data
        assert data["id"] == chunk_id
        assert "document_id" in data
        assert data["document_id"] == document_id

    def test_get_all_chunks_includes_uuids(self):
        """Test that listing chunks includes UUIDs for all items."""
        # Create library and document
        library_response = client.post(
            "/api/v1/libraries",
            json={"name": "Test Library"},
        )
        library_id = library_response.json()["id"]

        document_response = client.post(
            f"/api/v1/documents?library_id={library_id}",
            json={"name": "Test Document"},
        )
        document_id = document_response.json()["id"]

        # Create multiple chunks
        chunk_ids = []
        for i in range(3):
            response = client.post(
                f"/api/v1/chunks?document_id={document_id}",
                json={
                    "text": f"Chunk {i}",
                    "embedding": [float(i), 0.0, 0.0],
                },
            )
            chunk_ids.append(response.json()["id"])

        # Get all chunks
        response = client.get(f"/api/v1/chunks?document_id={document_id}")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify all chunks have UUIDs
        assert len(data) == 3

        for item in data:
            assert "id" in item
            assert item["id"] is not None
            assert "document_id" in item
            assert item["document_id"] == document_id
            # Verify valid UUIDs
            UUID(item["id"])
            UUID(item["document_id"])
            # Verify it's one of our created chunks
            assert item["id"] in chunk_ids

    def test_update_chunk_returns_uuid(self):
        """Test that updating a chunk returns its UUID."""
        # Create library, document, and chunk
        library_response = client.post(
            "/api/v1/libraries",
            json={"name": "Test Library"},
        )
        library_id = library_response.json()["id"]

        document_response = client.post(
            f"/api/v1/documents?library_id={library_id}",
            json={"name": "Test Document"},
        )
        document_id = document_response.json()["id"]

        create_response = client.post(
            f"/api/v1/chunks?document_id={document_id}",
            json={
                "text": "Original text",
                "embedding": [1.0, 2.0, 3.0],
            },
        )
        chunk_id = create_response.json()["id"]

        # Update the chunk
        response = client.put(
            f"/api/v1/chunks/{chunk_id}",
            json={"text": "Updated text"},
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify UUIDs are present and unchanged
        assert "id" in data
        assert data["id"] == chunk_id
        assert "document_id" in data
        assert data["document_id"] == document_id


class TestSearchUUIDs:
    """Test that search endpoint includes UUIDs."""

    def test_search_results_include_chunk_uuids(self):
        """Test that search results include chunk UUIDs."""
        # Create library
        library_response = client.post(
            "/api/v1/libraries",
            json={"name": "Test Library"},
        )
        library_id = library_response.json()["id"]

        # Create document
        document_response = client.post(
            f"/api/v1/documents?library_id={library_id}",
            json={"name": "Test Document"},
        )
        document_id = document_response.json()["id"]

        # Create chunks
        chunk_ids = []
        for i in range(5):
            response = client.post(
                f"/api/v1/chunks?document_id={document_id}",
                json={
                    "text": f"Chunk {i}",
                    "embedding": [float(i), 0.0, 0.0],
                },
            )
            chunk_ids.append(response.json()["id"])

        # Perform search
        response = client.post(
            f"/api/v1/libraries/{library_id}/search",
            json={
                "embedding": [0.0, 0.0, 0.0],
                "k": 3,
            },
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify response structure
        assert "results" in data
        assert "library_id" in data
        assert data["library_id"] == library_id

        # Verify each result includes chunk UUIDs
        for result in data["results"]:
            assert "chunk" in result
            chunk = result["chunk"]

            assert "id" in chunk
            assert chunk["id"] is not None
            assert "document_id" in chunk
            assert chunk["document_id"] == document_id

            # Verify valid UUIDs
            UUID(chunk["id"])
            UUID(chunk["document_id"])

            # Verify it's one of our chunks
            assert chunk["id"] in chunk_ids


class TestNestedEntityUUIDs:
    """Test that nested entities include UUIDs."""

    def test_library_documents_include_uuids(self):
        """Test that documents nested in library response include UUIDs."""
        # Create library with document
        library_response = client.post(
            "/api/v1/libraries",
            json={
                "name": "Test Library",
                "documents": [
                    {
                        "name": "Test Document",
                    }
                ],
            },
        )
        assert library_response.status_code == status.HTTP_201_CREATED
        data = library_response.json()

        # Verify library has UUID
        assert "id" in data
        library_id = data["id"]
        UUID(library_id)

        # Verify nested documents have UUIDs
        assert "documents" in data
        if len(data["documents"]) > 0:
            for doc in data["documents"]:
                assert "id" in doc
                assert doc["id"] is not None
                assert "library_id" in doc
                assert doc["library_id"] == library_id
                UUID(doc["id"])

    def test_document_chunks_include_uuids(self):
        """Test that chunks nested in document response include UUIDs."""
        # Create library
        library_response = client.post(
            "/api/v1/libraries",
            json={"name": "Test Library"},
        )
        library_id = library_response.json()["id"]

        # Create document with chunks
        document_response = client.post(
            f"/api/v1/documents?library_id={library_id}",
            json={
                "name": "Test Document",
                "chunks": [
                    {
                        "text": "Chunk 1",
                        "embedding": [1.0, 0.0, 0.0],
                    },
                    {
                        "text": "Chunk 2",
                        "embedding": [0.0, 1.0, 0.0],
                    },
                ],
            },
        )
        assert document_response.status_code == status.HTTP_201_CREATED
        data = document_response.json()

        # Verify document has UUID
        assert "id" in data
        document_id = data["id"]
        UUID(document_id)

        # Verify nested chunks have UUIDs
        assert "chunks" in data
        if len(data["chunks"]) > 0:
            for chunk in data["chunks"]:
                assert "id" in chunk
                assert chunk["id"] is not None
                assert "document_id" in chunk
                assert chunk["document_id"] == document_id
                UUID(chunk["id"])
