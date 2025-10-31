"""Tests for index storage module."""

import pytest
from pathlib import Path
import tempfile
import shutil

from vector_db.core.persistence.index_storage import IndexStorage
from vector_db.indexes import FlatIndex, DistanceMetric


@pytest.fixture
def temp_indexes_dir():
    """Create a temporary indexes directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def index_storage(temp_indexes_dir):
    """Create an IndexStorage instance."""
    return IndexStorage(temp_indexes_dir)


def test_index_storage_initialization(temp_indexes_dir):
    """Test index storage creates directory."""
    storage = IndexStorage(temp_indexes_dir / "subdir")
    assert (temp_indexes_dir / "subdir").exists()


def test_save_and_load_index(index_storage):
    """Test saving and loading an index."""
    library_id = "test-library-123"
    index = FlatIndex(distance_metric=DistanceMetric.COSINE)

    # Save index
    index_storage.save(library_id, index)

    # Load index
    loaded = index_storage.load(library_id)

    assert loaded is not None
    assert isinstance(loaded, FlatIndex)
    assert loaded.distance_metric == DistanceMetric.COSINE


def test_load_nonexistent_index(index_storage):
    """Test loading index that doesn't exist returns None."""
    result = index_storage.load("nonexistent-library")
    assert result is None


def test_load_corrupted_index(index_storage, temp_indexes_dir):
    """Test loading corrupted index file returns None."""
    library_id = "test-library-corrupted"

    # Create a corrupted pickle file
    corrupted_path = temp_indexes_dir / f"{library_id}.pkl"
    with open(corrupted_path, "wb") as f:
        f.write(b"corrupted data that's not a valid pickle")

    # Should return None for corrupted file
    result = index_storage.load(library_id)
    assert result is None


def test_delete_index(index_storage):
    """Test deleting an index."""
    library_id = "test-library-456"
    index = FlatIndex(distance_metric=DistanceMetric.EUCLIDEAN)

    # Save and verify
    index_storage.save(library_id, index)
    assert index_storage.exists(library_id)

    # Delete
    index_storage.delete(library_id)
    assert not index_storage.exists(library_id)
    assert index_storage.load(library_id) is None


def test_exists(index_storage):
    """Test checking if index exists."""
    library_id = "test-library-789"

    assert not index_storage.exists(library_id)

    index = FlatIndex(distance_metric=DistanceMetric.DOT_PRODUCT)
    index_storage.save(library_id, index)

    assert index_storage.exists(library_id)


def test_atomic_write(index_storage):
    """Test that saves are atomic (using temp file)."""
    library_id = "test-library-atomic"
    index1 = FlatIndex(distance_metric=DistanceMetric.COSINE)
    index2 = FlatIndex(distance_metric=DistanceMetric.EUCLIDEAN)

    # Save first version
    index_storage.save(library_id, index1)

    # Save second version (should replace atomically)
    index_storage.save(library_id, index2)

    # Should have the second version
    loaded = index_storage.load(library_id)
    assert loaded.distance_metric == DistanceMetric.EUCLIDEAN


def test_save_index_with_data(index_storage):
    """Test saving and loading index with actual data."""
    from vector_db.models import Chunk
    from uuid import uuid4
    from datetime import datetime

    library_id = "test-library-with-data"
    index = FlatIndex(distance_metric=DistanceMetric.COSINE)

    # Add some chunks to the index
    for i in range(5):
        chunk = Chunk(
            id=uuid4(),
            document_id=uuid4(),
            text=f"Test chunk {i}",
            embedding=[float(j) for j in range(128)],
            metadata={"index": i},
            created_at=datetime.utcnow(),
        )
        index.add(chunk)

    # Save index
    index_storage.save(library_id, index)

    # Load and verify
    loaded = index_storage.load(library_id)
    assert loaded is not None
    assert isinstance(loaded, FlatIndex)

    # Verify data is preserved
    results = loaded.search(
        query_embedding=[0.0] * 128,
        k=5,
    )
    assert len(results) == 5
