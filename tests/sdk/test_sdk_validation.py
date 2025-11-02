"""Tests for SDK validation of invalid index configurations."""

import pytest

from vector_db.sdk import VectorDBClient
from vector_db.sdk.exceptions import ValidationError


@pytest.fixture
def sdk_client():
    """Create SDK client."""
    client = VectorDBClient(base_url="http://localhost:8000")
    yield client
    client.close()


def test_invalid_index_type(sdk_client):
    """Test that invalid index type is rejected."""
    with pytest.raises((ValidationError, ValueError)) as exc_info:
        sdk_client.create_library(
            name="Test Invalid Index Type",
            index_type="invalid_type",
            distance_metric="cosine"
        )
    assert "index_type" in str(exc_info.value).lower() or "literal" in str(exc_info.value).lower()


def test_invalid_distance_metric(sdk_client):
    """Test that invalid distance metric is rejected."""
    with pytest.raises((ValidationError, ValueError)) as exc_info:
        sdk_client.create_library(
            name="Test Invalid Distance Metric",
            index_type="flat",
            distance_metric="invalid_metric"
        )
    assert "distance_metric" in str(exc_info.value).lower() or "literal" in str(exc_info.value).lower()


def test_lsh_params_on_flat_index(sdk_client):
    """Test that LSH parameters are rejected on flat index."""
    with pytest.raises((ValidationError, ValueError)) as exc_info:
        sdk_client.create_library(
            name="Test LSH Params on Flat",
            index_type="flat",
            distance_metric="cosine",
            n_hash_tables=5
        )
    assert "lsh" in str(exc_info.value).lower()


def test_lsh_params_on_hnsw_index(sdk_client):
    """Test that LSH parameters are rejected on HNSW index."""
    with pytest.raises((ValidationError, ValueError)) as exc_info:
        sdk_client.create_library(
            name="Test LSH Params on HNSW",
            index_type="hnsw",
            distance_metric="cosine",
            n_hash_bits=8
        )
    assert "lsh" in str(exc_info.value).lower()


def test_hnsw_params_on_flat_index(sdk_client):
    """Test that HNSW parameters are rejected on flat index."""
    with pytest.raises((ValidationError, ValueError)) as exc_info:
        sdk_client.create_library(
            name="Test HNSW Params on Flat",
            index_type="flat",
            distance_metric="cosine",
            M=16
        )
    assert "hnsw" in str(exc_info.value).lower()


def test_hnsw_params_on_lsh_index(sdk_client):
    """Test that HNSW parameters are rejected on LSH index."""
    with pytest.raises((ValidationError, ValueError)) as exc_info:
        sdk_client.create_library(
            name="Test HNSW Params on LSH",
            index_type="lsh",
            distance_metric="cosine",
            ef_construction=200
        )
    assert "hnsw" in str(exc_info.value).lower()


def test_negative_lsh_params(sdk_client):
    """Test that negative LSH parameters are rejected."""
    with pytest.raises((ValidationError, ValueError)) as exc_info:
        sdk_client.create_library(
            name="Test Negative LSH Params",
            index_type="lsh",
            distance_metric="cosine",
            n_hash_tables=-5
        )
    assert "greater than 0" in str(exc_info.value).lower() or "validation" in str(exc_info.value).lower()


def test_negative_hnsw_params(sdk_client):
    """Test that negative HNSW parameters are rejected."""
    with pytest.raises((ValidationError, ValueError)) as exc_info:
        sdk_client.create_library(
            name="Test Negative HNSW Params",
            index_type="hnsw",
            distance_metric="cosine",
            M=-16
        )
    assert "greater than 0" in str(exc_info.value).lower() or "validation" in str(exc_info.value).lower()


def test_zero_lsh_params(sdk_client):
    """Test that zero LSH parameters are rejected."""
    with pytest.raises((ValidationError, ValueError)) as exc_info:
        sdk_client.create_library(
            name="Test Zero LSH Params",
            index_type="lsh",
            distance_metric="cosine",
            n_hash_tables=0
        )
    assert "greater than 0" in str(exc_info.value).lower() or "validation" in str(exc_info.value).lower()


def test_zero_hnsw_params(sdk_client):
    """Test that zero HNSW parameters are rejected."""
    with pytest.raises((ValidationError, ValueError)) as exc_info:
        sdk_client.create_library(
            name="Test Zero HNSW Params",
            index_type="hnsw",
            distance_metric="cosine",
            ef_construction=0
        )
    assert "greater than 0" in str(exc_info.value).lower() or "validation" in str(exc_info.value).lower()
