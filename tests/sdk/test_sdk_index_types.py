"""Tests for SDK index type configuration."""

import pytest

from vector_db.sdk import VectorDBClient


@pytest.fixture
def sdk_client():
    """Create SDK client."""
    client = VectorDBClient(base_url="http://localhost:8000")
    yield client
    client.close()


def test_create_flat_index(sdk_client):
    """Test creating library with Flat index."""
    library = sdk_client.create_library(
        name="Test Flat Library",
        index_type="flat",
        distance_metric="cosine"
    )

    try:
        assert library.index_config.index_type == "flat"
        assert library.index_config.distance_metric == "cosine"
    finally:
        sdk_client.delete_library(library.id)


def test_create_lsh_index_with_defaults(sdk_client):
    """Test creating library with LSH index using defaults."""
    library = sdk_client.create_library(
        name="Test LSH Library Default",
        index_type="lsh",
        distance_metric="cosine"
    )

    try:
        assert library.index_config.index_type == "lsh"
        assert library.index_config.distance_metric == "cosine"
        # Defaults are set on backend but API returns None if not explicitly set
        assert library.index_config.n_hash_tables is None
        assert library.index_config.n_hash_bits is None
    finally:
        sdk_client.delete_library(library.id)


def test_create_lsh_index_with_custom_params(sdk_client):
    """Test creating library with LSH index and custom parameters."""
    library = sdk_client.create_library(
        name="Test LSH Library Custom",
        index_type="lsh",
        distance_metric="cosine",
        n_hash_tables=10,
        n_hash_bits=7
    )

    try:
        assert library.index_config.index_type == "lsh"
        assert library.index_config.distance_metric == "cosine"
        assert library.index_config.n_hash_tables == 10
        assert library.index_config.n_hash_bits == 7
    finally:
        sdk_client.delete_library(library.id)


def test_create_hnsw_index_with_defaults(sdk_client):
    """Test creating library with HNSW index using defaults."""
    library = sdk_client.create_library(
        name="Test HNSW Library Default",
        index_type="hnsw",
        distance_metric="cosine"
    )

    try:
        assert library.index_config.index_type == "hnsw"
        assert library.index_config.distance_metric == "cosine"
        # Defaults are set on backend but API returns None if not explicitly set
        assert library.index_config.M is None
        assert library.index_config.ef_construction is None
        assert library.index_config.ef_search is None
    finally:
        sdk_client.delete_library(library.id)


def test_create_hnsw_index_with_custom_params(sdk_client):
    """Test creating library with HNSW index and custom parameters."""
    library = sdk_client.create_library(
        name="Test HNSW Library Custom",
        index_type="hnsw",
        distance_metric="cosine",
        M=32,
        ef_construction=400,
        ef_search=150
    )

    try:
        assert library.index_config.index_type == "hnsw"
        assert library.index_config.distance_metric == "cosine"
        assert library.index_config.M == 32
        assert library.index_config.ef_construction == 400
        assert library.index_config.ef_search == 150
    finally:
        sdk_client.delete_library(library.id)


def test_create_library_with_euclidean_distance(sdk_client):
    """Test creating library with euclidean distance metric."""
    library = sdk_client.create_library(
        name="Test Euclidean Library",
        index_type="flat",
        distance_metric="euclidean"
    )

    try:
        assert library.index_config.index_type == "flat"
        assert library.index_config.distance_metric == "euclidean"
    finally:
        sdk_client.delete_library(library.id)


def test_create_library_with_dot_product_distance(sdk_client):
    """Test creating library with dot product distance metric."""
    library = sdk_client.create_library(
        name="Test Dot Product Library",
        index_type="flat",
        distance_metric="dot_product"
    )

    try:
        assert library.index_config.index_type == "flat"
        assert library.index_config.distance_metric == "dot_product"
    finally:
        sdk_client.delete_library(library.id)


def test_create_library_with_metadata(sdk_client):
    """Test creating library with custom metadata."""
    library = sdk_client.create_library(
        name="Test Library with Metadata",
        index_type="flat",
        distance_metric="cosine",
        metadata={
            "description": "Test library",
            "owner": "test@example.com",
            "category": "test",
            "tags": ["test", "example"],
            "custom": {"version": "1.0", "status": "active"}
        }
    )

    try:
        assert library.metadata.description == "Test library"
        assert library.metadata.owner == "test@example.com"
        assert library.metadata.category == "test"
        assert library.metadata.tags == ["test", "example"]
        assert library.metadata.custom == {"version": "1.0", "status": "active"}
    finally:
        sdk_client.delete_library(library.id)


def test_create_library_with_index_config_dict(sdk_client):
    """Test creating library with complete index_config dict."""
    index_config = {
        "index_type": "hnsw",
        "distance_metric": "cosine",
        "M": 24,
        "ef_construction": 300,
        "ef_search": 100
    }

    library = sdk_client.create_library(
        name="Test Library with Config Dict",
        index_config=index_config
    )

    try:
        assert library.index_config.index_type == "hnsw"
        assert library.index_config.distance_metric == "cosine"
        assert library.index_config.M == 24
        assert library.index_config.ef_construction == 300
        assert library.index_config.ef_search == 100
    finally:
        sdk_client.delete_library(library.id)


