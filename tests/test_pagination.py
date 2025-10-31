"""Tests for pagination."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

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


class TestLibraryPagination:
    """Tests for library pagination."""

    def test_library_pagination_defaults(self):
        """Test library pagination with default parameters."""
        # Create 3 libraries
        for i in range(3):
            client.post(
                "/api/v1/libraries",
                json={"name": f"Library {i}"},
            )

        # Get with defaults (skip=0, limit=50)
        response = client.get("/api/v1/libraries")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["items"]) == 3
        assert data["total"] == 3
        assert data["skip"] == 0
        assert data["limit"] == 50
        assert data["has_more"] is False

    def test_library_pagination(self):
        """Test library pagination with custom parameters."""
        # Create 5 libraries
        for i in range(5):
            client.post(
                "/api/v1/libraries",
                json={"name": f"Library {i}"},
            )

        # Get first page
        response = client.get("/api/v1/libraries?skip=0&limit=2")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["items"]) == 2
        assert data["total"] == 5
        assert data["skip"] == 0
        assert data["limit"] == 2
        assert data["has_more"] is True

        # Get second page
        response = client.get("/api/v1/libraries?skip=2&limit=2")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["items"]) == 2
        assert data["total"] == 5
        assert data["skip"] == 2
        assert data["has_more"] is True

        # Get last page
        response = client.get("/api/v1/libraries?skip=4&limit=2")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["items"]) == 1
        assert data["total"] == 5
        assert data["has_more"] is False



class TestDocumentPagination:
    """Tests for document pagination."""

    def test_document_pagination(self):
        """Test document pagination."""
        # Create a library
        library_response = client.post(
            "/api/v1/libraries",
            json={"name": "Test Library"},
        )
        library_id = library_response.json()["id"]

        # Create 5 documents
        for i in range(5):
            client.post(
                f"/api/v1/documents?library_id={library_id}",
                json={"name": f"Document {i}"},
            )

        # Get first page
        response = client.get(
            f"/api/v1/documents?library_id={library_id}&skip=0&limit=2"
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["items"]) == 2
        assert data["total"] == 5
        assert data["has_more"] is True
