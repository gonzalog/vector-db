"""Tests for API endpoints."""

from fastapi import status


class TestLibraryEndpoints:
    """Tests for library CRUD endpoints."""

    def test_create_library(self, client):
        """Test creating a library."""
        response = client.post(
            "/api/v1/libraries",
            json={"name": "Test Library", "metadata": {"description": "Test"}},
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["name"] == "Test Library"
        assert data["metadata"]["description"] == "Test"
        assert "id" in data

    def test_get_all_libraries(self, client):
        """Test getting all libraries."""
        # Create two libraries
        client.post(
            "/api/v1/libraries",
            json={"name": "Library 1"},
        )
        client.post(
            "/api/v1/libraries",
            json={"name": "Library 2"},
        )

        response = client.get("/api/v1/libraries")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "items" in data
        assert len(data["items"]) == 2
        assert data["total"] == 2

    def test_get_library_by_id(self, client):
        """Test getting a library by ID."""
        create_response = client.post(
            "/api/v1/libraries",
            json={"name": "Test Library"},
        )
        library_id = create_response.json()["id"]

        response = client.get(f"/api/v1/libraries/{library_id}")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == library_id
        assert data["name"] == "Test Library"

    def test_get_library_not_found(self, client):
        """Test getting a non-existent library."""
        response = client.get(
            "/api/v1/libraries/00000000-0000-0000-0000-000000000000"
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_update_library(self, client):
        """Test updating a library."""
        create_response = client.post(
            "/api/v1/libraries",
            json={"name": "Old Name"},
        )
        library_id = create_response.json()["id"]

        response = client.put(
            f"/api/v1/libraries/{library_id}",
            json={"name": "New Name"},
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == "New Name"

    def test_delete_library(self, client):
        """Test deleting a library."""
        create_response = client.post(
            "/api/v1/libraries",
            json={"name": "Test Library"},
        )
        library_id = create_response.json()["id"]

        response = client.delete(f"/api/v1/libraries/{library_id}")
        assert response.status_code == status.HTTP_204_NO_CONTENT

        # Verify it's deleted
        get_response = client.get(f"/api/v1/libraries/{library_id}")
        assert get_response.status_code == status.HTTP_404_NOT_FOUND


class TestDocumentEndpoints:
    """Tests for document CRUD endpoints."""

    def test_create_document(self, client):
        """Test creating a document."""
        # Create a library first
        library_response = client.post(
            "/api/v1/libraries",
            json={"name": "Test Library"},
        )
        library_id = library_response.json()["id"]

        response = client.post(
            f"/api/v1/documents?library_id={library_id}",
            json={"name": "Test Document", "metadata": {"title": "Test"}},
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["name"] == "Test Document"
        assert data["library_id"] == library_id

    def test_get_document_by_id(self, client):
        """Test getting a document by ID."""
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

        response = client.get(f"/api/v1/documents/{document_id}")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == document_id

    def test_update_document(self, client):
        """Test updating a document."""
        # Create library and document
        library_response = client.post(
            "/api/v1/libraries",
            json={"name": "Test Library"},
        )
        library_id = library_response.json()["id"]

        create_response = client.post(
            f"/api/v1/documents?library_id={library_id}",
            json={"name": "Old Name"},
        )
        document_id = create_response.json()["id"]

        response = client.put(
            f"/api/v1/documents/{document_id}",
            json={"name": "New Name"},
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == "New Name"

    def test_delete_document(self, client):
        """Test deleting a document."""
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

        response = client.delete(f"/api/v1/documents/{document_id}")
        assert response.status_code == status.HTTP_204_NO_CONTENT


class TestChunkEndpoints:
    """Tests for chunk CRUD endpoints."""

    def test_create_chunk(self, client):
        """Test creating a chunk."""
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

        response = client.post(
            f"/api/v1/chunks?document_id={document_id}",
            json={
                "text": "Test chunk text",
                "embedding": [0.1, 0.2, 0.3],
            },
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["text"] == "Test chunk text"
        assert data["document_id"] == document_id

    def test_get_chunk_by_id(self, client):
        """Test getting a chunk by ID."""
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
            json={"text": "Test chunk", "embedding": [0.1, 0.2, 0.3]},
        )
        chunk_id = create_response.json()["id"]

        response = client.get(f"/api/v1/chunks/{chunk_id}")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == chunk_id

    def test_update_chunk(self, client):
        """Test updating a chunk."""
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
            json={"text": "Old text", "embedding": [0.1, 0.2, 0.3]},
        )
        chunk_id = create_response.json()["id"]

        response = client.put(
            f"/api/v1/chunks/{chunk_id}",
            json={"text": "New text"},
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["text"] == "New text"

    def test_delete_chunk(self, client):
        """Test deleting a chunk."""
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
            json={"text": "Test chunk", "embedding": [0.1, 0.2, 0.3]},
        )
        chunk_id = create_response.json()["id"]

        response = client.delete(f"/api/v1/chunks/{chunk_id}")
        assert response.status_code == status.HTTP_204_NO_CONTENT
