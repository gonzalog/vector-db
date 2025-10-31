"""Tests for index configuration in libraries."""

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
    """Clear all repositories and indexes before each test."""
    library_repository.clear()
    document_repository.clear()
    chunk_repository.clear()
    yield
    library_repository.clear()
    document_repository.clear()
    chunk_repository.clear()


class TestIndexConfiguration:
    """Tests for index configuration in libraries."""

    def test_create_library_with_default_index_config(self):
        """Test creating a library with default index configuration."""
        response = client.post(
            "/api/v1/libraries",
            json={"name": "Test Library"},
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()

        # Verify default index config
        assert "index_config" in data
        assert data["index_config"]["index_type"] == "flat"
        assert data["index_config"]["distance_metric"] == "cosine"

    def test_create_library_with_flat_index(self):
        """Test creating a library with flat index configuration."""
        response = client.post(
            "/api/v1/libraries",
            json={
                "name": "Flat Library",
                "index_config": {
                    "index_type": "flat",
                    "distance_metric": "euclidean",
                },
            },
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()

        assert data["index_config"]["index_type"] == "flat"
        assert data["index_config"]["distance_metric"] == "euclidean"

    def test_create_library_with_lsh_index(self):
        """Test creating a library with LSH index configuration."""
        response = client.post(
            "/api/v1/libraries",
            json={
                "name": "LSH Library",
                "index_config": {
                    "index_type": "lsh",
                    "distance_metric": "cosine",
                    "n_hash_tables": 5,
                    "n_hash_bits": 8,
                },
            },
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()

        assert data["index_config"]["index_type"] == "lsh"
        assert data["index_config"]["distance_metric"] == "cosine"
        assert data["index_config"]["n_hash_tables"] == 5
        assert data["index_config"]["n_hash_bits"] == 8

    def test_create_library_with_hnsw_index(self):
        """Test creating a library with HNSW index configuration."""
        response = client.post(
            "/api/v1/libraries",
            json={
                "name": "HNSW Library",
                "index_config": {
                    "index_type": "hnsw",
                    "distance_metric": "cosine",
                    "M": 16,
                    "ef_construction": 200,
                    "ef_search": 50,
                },
            },
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()

        assert data["index_config"]["index_type"] == "hnsw"
        assert data["index_config"]["distance_metric"] == "cosine"
        assert data["index_config"]["M"] == 16
        assert data["index_config"]["ef_construction"] == 200
        assert data["index_config"]["ef_search"] == 50

    def test_index_created_on_library_creation(self):
        """Test that index is created when library is created."""
        # Create library
        response = client.post(
            "/api/v1/libraries",
            json={"name": "Test Library"},
        )

        library_id = response.json()["id"]

        # Verify index works by searching (even if empty)
        search_response = client.post(
            f"/api/v1/libraries/{library_id}/search",
            json={"embedding": [1.0, 0.0, 0.0], "k": 5},
        )

        assert search_response.status_code == status.HTTP_200_OK
        data = search_response.json()
        assert "results" in data
        assert data["total_results"] == 0  # Empty library

    def test_index_updated_on_chunk_add(self):
        """Test that index is updated when chunks are added."""
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
        for i in range(3):
            client.post(
                f"/api/v1/chunks?document_id={document_id}",
                json={
                    "text": f"Test chunk {i}",
                    "embedding": [float(i), 1.0, 0.0],
                },
            )

        # Verify index contains chunks by searching
        search_response = client.post(
            f"/api/v1/libraries/{library_id}/search",
            json={"embedding": [0.0, 1.0, 0.0], "k": 2},
        )

        assert search_response.status_code == status.HTTP_200_OK
        data = search_response.json()
        assert data["total_results"] == 2
        assert len(data["results"]) == 2

    def test_index_updated_on_chunk_delete(self):
        """Test that index is updated when chunks are deleted."""
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
        for i in range(3):
            chunk_response = client.post(
                f"/api/v1/chunks?document_id={document_id}",
                json={
                    "text": f"Test chunk {i}",
                    "embedding": [float(i), 1.0, 0.0],
                },
            )
            chunk_ids.append(chunk_response.json()["id"])

        # Verify index has 3 chunks by searching
        search_response = client.post(
            f"/api/v1/libraries/{library_id}/search",
            json={"embedding": [0.0, 1.0, 0.0], "k": 10},
        )
        assert search_response.status_code == status.HTTP_200_OK
        assert search_response.json()["total_results"] == 3

        # Delete one chunk
        delete_response = client.delete(f"/api/v1/chunks/{chunk_ids[0]}")
        assert delete_response.status_code == status.HTTP_204_NO_CONTENT

        # Verify index has 2 chunks by searching again
        search_response = client.post(
            f"/api/v1/libraries/{library_id}/search",
            json={"embedding": [0.0, 1.0, 0.0], "k": 10},
        )
        assert search_response.status_code == status.HTTP_200_OK
        assert search_response.json()["total_results"] == 2

    def test_index_deleted_on_library_delete(self):
        """Test that index is deleted when library is deleted."""
        # Create library
        library_response = client.post(
            "/api/v1/libraries",
            json={"name": "Test Library"},
        )
        library_id = library_response.json()["id"]

        # Verify search works (index exists)
        search_response = client.post(
            f"/api/v1/libraries/{library_id}/search",
            json={"embedding": [1.0, 0.0, 0.0], "k": 5},
        )
        assert search_response.status_code == status.HTTP_200_OK

        # Delete library
        delete_response = client.delete(f"/api/v1/libraries/{library_id}")
        assert delete_response.status_code == status.HTTP_204_NO_CONTENT

        # Verify searching deleted library fails
        search_response = client.post(
            f"/api/v1/libraries/{library_id}/search",
            json={"embedding": [1.0, 0.0, 0.0], "k": 5},
        )
        assert search_response.status_code == status.HTTP_404_NOT_FOUND

    def test_search_uses_library_index_config(self):
        """Test that search uses the library's configured index."""
        # Create library with HNSW index
        library_response = client.post(
            "/api/v1/libraries",
            json={
                "name": "HNSW Library",
                "index_config": {
                    "index_type": "hnsw",
                    "distance_metric": "euclidean",
                    "M": 8,
                    "ef_construction": 100,
                    "ef_search": 50,
                },
            },
        )
        library_id = library_response.json()["id"]

        # Create document and chunks
        document_response = client.post(
            f"/api/v1/documents?library_id={library_id}",
            json={"name": "Test Document"},
        )
        document_id = document_response.json()["id"]

        for i in range(10):
            client.post(
                f"/api/v1/chunks?document_id={document_id}",
                json={
                    "text": f"Test chunk {i}",
                    "embedding": [float(i), 0.0, 0.0],
                },
            )

        # Search should work with HNSW index
        search_response = client.post(
            f"/api/v1/libraries/{library_id}/search",
            json={
                "embedding": [0.0, 0.0, 0.0],
                "k": 3,
            },
        )

        assert search_response.status_code == status.HTTP_200_OK
        data = search_response.json()
        assert len(data["results"]) <= 3
