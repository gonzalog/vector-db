"""Tests for vector indexes."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from vector_db.indexes import DistanceMetric, FlatIndex, HNSWIndex, LSHIndex
from vector_db.main import app
from vector_db.models import Chunk
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


class TestDistanceMetrics:
    """Tests for distance metric functions."""

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        from vector_db.indexes.distance import cosine_similarity

        # Identical vectors
        assert cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)

        # Orthogonal vectors
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

        # Opposite vectors
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        from vector_db.indexes.distance import euclidean_distance

        # Identical vectors
        assert euclidean_distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)

        # Simple distance
        assert euclidean_distance([0.0, 0.0], [3.0, 4.0]) == pytest.approx(5.0)


class TestFlatIndex:
    """Tests for flat (brute-force) index."""

    def test_add_and_search(self):
        """Test adding chunks and searching."""
        index = FlatIndex()

        # Create test chunks
        chunk1 = Chunk(
            document_id="00000000-0000-0000-0000-000000000001",
            text="test 1",
            embedding=[1.0, 0.0, 0.0],
        )
        chunk2 = Chunk(
            document_id="00000000-0000-0000-0000-000000000001",
            text="test 2",
            embedding=[0.0, 1.0, 0.0],
        )
        chunk3 = Chunk(
            document_id="00000000-0000-0000-0000-000000000001",
            text="test 3",
            embedding=[0.0, 0.0, 1.0],
        )

        # Add chunks
        index.add(chunk1)
        index.add(chunk2)
        index.add(chunk3)

        assert index.size() == 3

        # Search for vector closest to chunk1
        results = index.search([1.0, 0.0, 0.0], k=2)

        assert len(results) == 2
        assert results[0][0].id == chunk1.id  # Closest match
        assert results[0][1] < results[1][1]  # Distance increases

    def test_add_batch(self):
        """Test batch addition."""
        index = FlatIndex()

        chunks = [
            Chunk(
                document_id="00000000-0000-0000-0000-000000000001",
                text=f"test {i}",
                embedding=[float(i), 0.0, 0.0],
            )
            for i in range(10)
        ]

        index.add_batch(chunks)
        assert index.size() == 10

    def test_metadata_filtering(self):
        """Test search with metadata filtering."""
        index = FlatIndex()

        chunk1 = Chunk(
            document_id="00000000-0000-0000-0000-000000000001",
            text="test 1",
            embedding=[1.0, 0.0, 0.0],
            metadata={"tags": ["important"], "author": "Alice"},
        )
        chunk2 = Chunk(
            document_id="00000000-0000-0000-0000-000000000001",
            text="test 2",
            embedding=[0.9, 0.1, 0.0],
            metadata={"tags": ["draft"], "author": "Bob"},
        )

        index.add(chunk1)
        index.add(chunk2)

        # Filter by author
        results = index.search([1.0, 0.0, 0.0], k=10, metadata_filter={"author": "Alice"})

        assert len(results) == 1
        assert results[0][0].id == chunk1.id

        # Filter by tags
        results = index.search(
            [1.0, 0.0, 0.0], k=10, metadata_filter={"tags": ["important"]}
        )

        assert len(results) == 1
        assert results[0][0].id == chunk1.id

    def test_remove_chunk(self):
        """Test removing chunks from index."""
        index = FlatIndex()

        chunk = Chunk(
            document_id="00000000-0000-0000-0000-000000000001",
            text="test",
            embedding=[1.0, 0.0, 0.0],
        )

        index.add(chunk)
        assert index.size() == 1

        # Remove chunk
        removed = index.remove(chunk.id)
        assert removed is True
        assert index.size() == 0

        # Try removing again
        removed = index.remove(chunk.id)
        assert removed is False

    def test_clear(self):
        """Test clearing the index."""
        index = FlatIndex()

        chunks = [
            Chunk(
                document_id="00000000-0000-0000-0000-000000000001",
                text=f"test {i}",
                embedding=[float(i), 0.0, 0.0],
            )
            for i in range(5)
        ]

        index.add_batch(chunks)
        assert index.size() == 5

        index.clear()
        assert index.size() == 0


class TestLSHIndex:
    """Tests for LSH (Locality-Sensitive Hashing) index."""

    def test_add_and_search(self):
        """Test adding chunks and searching with LSH."""
        index = LSHIndex(n_hash_tables=3, n_hash_bits=4)

        # Create test chunks
        chunks = [
            Chunk(
                document_id="00000000-0000-0000-0000-000000000001",
                text=f"test {i}",
                embedding=[float(i), 0.0, 0.0],
            )
            for i in range(10)
        ]

        # Add chunks
        index.add_batch(chunks)
        assert index.size() == 10

        # Search
        results = index.search([0.0, 0.0, 0.0], k=3)

        assert len(results) <= 3
        # LSH is approximate, so we just check we got results
        assert len(results) > 0

    def test_metadata_filtering(self):
        """Test LSH search with metadata filtering."""
        from vector_db.models import ChunkMetadata

        index = LSHIndex(n_hash_tables=3, n_hash_bits=4)

        chunks = [
            Chunk(
                document_id="00000000-0000-0000-0000-000000000001",
                text=f"test {i}",
                embedding=[float(i), 0.0, 0.0],
                metadata=ChunkMetadata(custom={"category": "A" if i < 5 else "B"}),
            )
            for i in range(10)
        ]

        index.add_batch(chunks)

        # Filter by category
        results = index.search([2.0, 0.0, 0.0], k=10, metadata_filter={"category": "A"})

        # All results should match the filter
        if results:
            assert all(result[0].metadata.custom["category"] == "A" for result in results)


class TestHNSWIndex:
    """Tests for HNSW (Hierarchical Navigable Small World) index."""

    def test_add_and_search(self):
        """Test adding chunks and searching with HNSW."""
        index = HNSWIndex(M=8, ef_construction=100, ef_search=50)

        # Create test chunks
        chunks = [
            Chunk(
                document_id="00000000-0000-0000-0000-000000000001",
                text=f"test {i}",
                embedding=[float(i), 0.0, 0.0],
            )
            for i in range(10)
        ]

        # Add chunks
        index.add_batch(chunks)
        assert index.size() == 10

        # Search
        results = index.search([0.0, 0.0, 0.0], k=3)

        assert len(results) <= 3
        assert len(results) > 0  # HNSW is approximate, so we just check we got results

    def test_metadata_filtering(self):
        """Test HNSW search with metadata filtering."""
        from vector_db.models import ChunkMetadata

        index = HNSWIndex(M=8, ef_construction=100, ef_search=50)

        chunks = [
            Chunk(
                document_id="00000000-0000-0000-0000-000000000001",
                text=f"test {i}",
                embedding=[float(i), 0.0, 0.0],
                metadata=ChunkMetadata(custom={"category": "A" if i < 5 else "B"}),
            )
            for i in range(10)
        ]

        index.add_batch(chunks)

        # Filter by category
        results = index.search([2.0, 0.0, 0.0], k=10, metadata_filter={"category": "A"})

        assert all(result[0].metadata.custom["category"] == "A" for result in results)


class TestSearchAPI:
    """Tests for search API endpoint."""

    def test_search_endpoint(self):
        """Test vector search endpoint."""
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
        for i in range(5):
            client.post(
                f"/api/v1/chunks?document_id={document_id}",
                json={
                    "text": f"Test chunk {i}",
                    "embedding": [float(i), 0.0, 0.0],
                    "metadata": {"index": i},
                },
            )

        # Search
        search_response = client.post(
            f"/api/v1/libraries/{library_id}/search",
            json={
                "embedding": [0.0, 0.0, 0.0],
                "k": 3,
            },
        )

        assert search_response.status_code == status.HTTP_200_OK
        data = search_response.json()

        assert "results" in data
        assert len(data["results"]) <= 3
        assert data["total_results"] <= 3
        assert data["library_id"] == library_id

        # Verify results have required fields
        for result in data["results"]:
            assert "chunk" in result
            assert "score" in result
            assert "distance" in result

    def test_search_with_metadata_filter(self):
        """Test search with metadata filtering."""
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

        # Create chunks with different metadata
        for i in range(5):
            client.post(
                f"/api/v1/chunks?document_id={document_id}",
                json={
                    "text": f"Test chunk {i}",
                    "embedding": [float(i), 0.0, 0.0],
                    "metadata": {
                        "category": "A" if i < 3 else "B",
                        "tags": ["important"] if i % 2 == 0 else [],
                    },
                },
            )

        # Search with category filter
        search_response = client.post(
            f"/api/v1/libraries/{library_id}/search",
            json={
                "embedding": [0.0, 0.0, 0.0],
                "k": 10,
                "metadata_filter": {"category": "A"},
            },
        )

        assert search_response.status_code == status.HTTP_200_OK
        data = search_response.json()

        # Should only return chunks with category A
        assert len(data["results"]) == 3

    def test_search_empty_library(self):
        """Test searching an empty library."""
        # Create library
        library_response = client.post(
            "/api/v1/libraries",
            json={"name": "Empty Library"},
        )
        library_id = library_response.json()["id"]

        # Search empty library
        search_response = client.post(
            f"/api/v1/libraries/{library_id}/search",
            json={
                "embedding": [0.0, 0.0, 0.0],
                "k": 10,
            },
        )

        assert search_response.status_code == status.HTTP_200_OK
        data = search_response.json()

        assert len(data["results"]) == 0
        assert data["total_results"] == 0
