"""Tests for vector storage module."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from vector_db.core.persistence.vector_storage import VectorStorage


@pytest.fixture
def temp_vectors_dir():
    """Create a temporary vectors directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def vector_storage(temp_vectors_dir):
    """Create a VectorStorage instance."""
    return VectorStorage(temp_vectors_dir)


def test_vector_storage_initialization(temp_vectors_dir):
    """Test vector storage creates directory."""
    storage = VectorStorage(temp_vectors_dir / "subdir")
    assert (temp_vectors_dir / "subdir").exists()


def test_save_and_load_vectors(vector_storage):
    """Test saving and loading vectors."""
    library_id = "test-library-123"
    vectors = np.random.rand(10, 128).astype(np.float32)

    # Save vectors
    vector_storage.save(library_id, vectors)

    # Load vectors
    loaded = vector_storage.load(library_id)

    assert loaded is not None
    assert np.array_equal(loaded, vectors)


def test_load_nonexistent_vectors(vector_storage):
    """Test loading vectors that don't exist returns None."""
    result = vector_storage.load("nonexistent-library")
    assert result is None


def test_delete_vectors(vector_storage):
    """Test deleting vectors."""
    library_id = "test-library-456"
    vectors = np.random.rand(5, 64).astype(np.float32)

    # Save and verify
    vector_storage.save(library_id, vectors)
    assert vector_storage.exists(library_id)

    # Delete
    vector_storage.delete(library_id)
    assert not vector_storage.exists(library_id)
    assert vector_storage.load(library_id) is None


def test_exists(vector_storage):
    """Test checking if vectors exist."""
    library_id = "test-library-789"

    assert not vector_storage.exists(library_id)

    vectors = np.random.rand(3, 32).astype(np.float32)
    vector_storage.save(library_id, vectors)

    assert vector_storage.exists(library_id)


def test_atomic_write(vector_storage):
    """Test that saves are atomic (using temp file)."""
    library_id = "test-library-atomic"
    vectors1 = np.random.rand(5, 64).astype(np.float32)
    vectors2 = np.random.rand(10, 64).astype(np.float32)

    # Save first version
    vector_storage.save(library_id, vectors1)

    # Save second version (should replace atomically)
    vector_storage.save(library_id, vectors2)

    # Should have the second version
    loaded = vector_storage.load(library_id)
    assert np.array_equal(loaded, vectors2)
    assert loaded.shape == (10, 64)


def test_empty_vectors_array(vector_storage):
    """Test saving and loading empty vectors array."""
    library_id = "test-library-empty"
    vectors = np.empty((0, 0), dtype=np.float32)

    vector_storage.save(library_id, vectors)
    loaded = vector_storage.load(library_id)

    assert loaded is not None
    assert loaded.shape == (0, 0)
