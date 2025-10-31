"""Integration tests for the complete persistence stack."""

import pytest
import pytest_asyncio
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
from uuid import uuid4

from vector_db.core.persistence.database import Database, init_database, close_database
from vector_db.core.persistence.vector_storage import VectorStorage
from vector_db.core.persistence.index_storage import IndexStorage
from vector_db.core.settings import Settings
from vector_db.models import Library, Document, Chunk, IndexConfig, VectorQuery
from vector_db.indexes import DistanceMetric


@pytest_asyncio.fixture
async def temp_persistence_env():
    """Create temporary environment for persistence testing."""
    temp_dir = tempfile.mkdtemp()

    # Setup settings
    settings = Settings()
    settings.DATA_DIR = Path(temp_dir)

    # Initialize database
    db = await init_database(settings.DATABASE_PATH)

    # Create storage managers
    vector_storage = VectorStorage(settings.VECTORS_DIR)
    index_storage = IndexStorage(settings.INDEXES_DIR)

    yield {
        "settings": settings,
        "db": db,
        "vector_storage": vector_storage,
        "index_storage": index_storage,
    }

    # Cleanup
    await close_database()
    shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_full_library_persistence_lifecycle(temp_persistence_env):
    """Test complete lifecycle: create library, add chunks, persist, restart, verify."""
    from vector_db.core.persistence.repositories import (
        LibraryRepository,
        DocumentRepository,
        ChunkRepository,
    )
    from vector_db.indexes import FlatIndex

    db = temp_persistence_env["db"]
    vector_storage = temp_persistence_env["vector_storage"]
    index_storage = temp_persistence_env["index_storage"]

    lib_repo = LibraryRepository(db)
    doc_repo = DocumentRepository(db)
    chunk_repo = ChunkRepository(db)

    # 1. Create library
    library = Library(
        id=uuid4(),
        name="Test Library",
        index_config=IndexConfig(
            index_type="flat",
            distance_metric="cosine",
        ),
        created_at=datetime.utcnow(),
    )
    await lib_repo.create(library)

    # 2. Create document
    document = Document(
        id=uuid4(),
        library_id=library.id,
        name="Test Document",
        created_at=datetime.utcnow(),
    )
    await doc_repo.create(document)

    # 3. Create chunks with embeddings
    num_chunks = 10
    embedding_dim = 128
    chunks = []
    vectors = []

    for i in range(num_chunks):
        embedding = np.random.rand(embedding_dim).astype(np.float32).tolist()
        chunk = Chunk(
            id=uuid4(),
            document_id=document.id,
            text=f"Test chunk {i}",
            embedding=embedding,
            metadata={"index": i},
            created_at=datetime.utcnow(),
        )
        chunks.append(chunk)
        vectors.append(embedding)
        await chunk_repo.create(chunk, vector_index=i)

    # 4. Save vectors
    vectors_array = np.array(vectors, dtype=np.float32)
    vector_storage.save(str(library.id), vectors_array)

    # 5. Create and save index
    index = FlatIndex(distance_metric=DistanceMetric.COSINE)
    for chunk in chunks:
        index.add(chunk)
    index_storage.save(str(library.id), index)

    # 6. Verify everything is persisted
    assert vector_storage.exists(str(library.id))
    assert index_storage.exists(str(library.id))

    # 7. "Restart" - reload from disk
    loaded_library = await lib_repo.get(str(library.id))
    assert loaded_library is not None
    assert loaded_library.name == library.name

    loaded_vectors = vector_storage.load(str(library.id))
    assert loaded_vectors is not None
    assert loaded_vectors.shape == (num_chunks, embedding_dim)

    loaded_index = index_storage.load(str(library.id))
    assert loaded_index is not None

    # 8. Test search works with reloaded index
    query_embedding = np.random.rand(embedding_dim).astype(np.float32).tolist()
    results = loaded_index.search(query_embedding=query_embedding, k=5)
    assert len(results) == 5


@pytest.mark.asyncio
async def test_library_deletion_cleanup(temp_persistence_env):
    """Test that deleting a library cleans up all associated data."""
    from vector_db.core.persistence.repositories import (
        LibraryRepository,
        DocumentRepository,
        ChunkRepository,
    )
    from vector_db.indexes import FlatIndex

    db = temp_persistence_env["db"]
    vector_storage = temp_persistence_env["vector_storage"]
    index_storage = temp_persistence_env["index_storage"]

    lib_repo = LibraryRepository(db)
    doc_repo = DocumentRepository(db)
    chunk_repo = ChunkRepository(db)

    # Create library with data
    library = Library(
        id=uuid4(),
        name="Library to Delete",
        index_config=IndexConfig(index_type="flat", distance_metric="cosine"),
        created_at=datetime.utcnow(),
    )
    await lib_repo.create(library)

    document = Document(
        id=uuid4(),
        library_id=library.id,
        name="Document",
        created_at=datetime.utcnow(),
    )
    await doc_repo.create(document)

    chunk = Chunk(
        id=uuid4(),
        document_id=document.id,
        text="Test",
        embedding=[0.1] * 128,  # Dummy embedding
        metadata={},
        created_at=datetime.utcnow(),
    )
    await chunk_repo.create(chunk, vector_index=0)

    # Save vectors and index
    vectors = np.random.rand(1, 128).astype(np.float32)
    vector_storage.save(str(library.id), vectors)

    index = FlatIndex(distance_metric=DistanceMetric.COSINE)
    index_storage.save(str(library.id), index)

    # Verify everything exists
    assert await lib_repo.get(str(library.id)) is not None
    assert await doc_repo.get(str(document.id)) is not None
    assert await chunk_repo.get(str(chunk.id)) is not None
    assert vector_storage.exists(str(library.id))
    assert index_storage.exists(str(library.id))

    # Delete library (should cascade)
    await lib_repo.delete(str(library.id))

    # Verify library and cascaded data are gone
    assert await lib_repo.get(str(library.id)) is None
    assert await doc_repo.get(str(document.id)) is None
    assert await chunk_repo.get(str(chunk.id)) is None

    # Note: Files must be deleted manually (not handled by DB cascade)
    # This would be done by PersistentLibraryRepository
    vector_storage.delete(str(library.id))
    index_storage.delete(str(library.id))

    assert not vector_storage.exists(str(library.id))
    assert not index_storage.exists(str(library.id))


@pytest.mark.asyncio
async def test_multiple_libraries_isolation(temp_persistence_env):
    """Test that multiple libraries don't interfere with each other."""
    from vector_db.core.persistence.repositories import LibraryRepository
    from vector_db.indexes import FlatIndex

    db = temp_persistence_env["db"]
    vector_storage = temp_persistence_env["vector_storage"]
    index_storage = temp_persistence_env["index_storage"]

    lib_repo = LibraryRepository(db)

    # Create two libraries
    library1 = Library(
        id=uuid4(),
        name="Library 1",
        index_config=IndexConfig(index_type="flat", distance_metric="cosine"),
        created_at=datetime.utcnow(),
    )
    await lib_repo.create(library1)

    library2 = Library(
        id=uuid4(),
        name="Library 2",
        index_config=IndexConfig(index_type="flat", distance_metric="euclidean"),
        created_at=datetime.utcnow(),
    )
    await lib_repo.create(library2)

    # Save different vectors for each
    vectors1 = np.random.rand(5, 64).astype(np.float32)
    vectors2 = np.random.rand(10, 128).astype(np.float32)

    vector_storage.save(str(library1.id), vectors1)
    vector_storage.save(str(library2.id), vectors2)

    # Save different indexes for each
    index1 = FlatIndex(distance_metric=DistanceMetric.COSINE)
    index2 = FlatIndex(distance_metric=DistanceMetric.EUCLIDEAN)

    index_storage.save(str(library1.id), index1)
    index_storage.save(str(library2.id), index2)

    # Verify isolation
    loaded_vectors1 = vector_storage.load(str(library1.id))
    loaded_vectors2 = vector_storage.load(str(library2.id))

    assert loaded_vectors1.shape == (5, 64)
    assert loaded_vectors2.shape == (10, 128)
    assert not np.array_equal(loaded_vectors1, loaded_vectors2[:5, :64])

    loaded_index1 = index_storage.load(str(library1.id))
    loaded_index2 = index_storage.load(str(library2.id))

    assert loaded_index1.distance_metric == DistanceMetric.COSINE
    assert loaded_index2.distance_metric == DistanceMetric.EUCLIDEAN


@pytest.mark.asyncio
async def test_concurrent_vector_updates(temp_persistence_env):
    """Test that vector array updates work correctly."""
    vector_storage = temp_persistence_env["vector_storage"]

    library_id = "test-concurrent"

    # Start with empty
    vectors = np.empty((0, 128), dtype=np.float32)
    vector_storage.save(library_id, vectors)

    # Simulate adding vectors one by one
    for i in range(10):
        new_vector = np.random.rand(1, 128).astype(np.float32)

        # Load current vectors
        current = vector_storage.load(library_id)

        # Append new vector
        if current.size == 0:
            updated = new_vector
        else:
            updated = np.vstack([current, new_vector])

        # Save updated array
        vector_storage.save(library_id, updated)

    # Verify final state
    final_vectors = vector_storage.load(library_id)
    assert final_vectors.shape == (10, 128)


@pytest.mark.asyncio
async def test_index_rebuild_from_vectors(temp_persistence_env):
    """Test rebuilding index from persisted vectors and chunk metadata."""
    from vector_db.core.persistence.repositories import (
        LibraryRepository,
        DocumentRepository,
        ChunkRepository,
    )
    from vector_db.indexes import FlatIndex

    db = temp_persistence_env["db"]
    vector_storage = temp_persistence_env["vector_storage"]
    index_storage = temp_persistence_env["index_storage"]

    lib_repo = LibraryRepository(db)
    doc_repo = DocumentRepository(db)
    chunk_repo = ChunkRepository(db)

    # Create library
    library = Library(
        id=uuid4(),
        name="Rebuild Test",
        index_config=IndexConfig(index_type="flat", distance_metric="cosine"),
        created_at=datetime.utcnow(),
    )
    await lib_repo.create(library)

    # Create document
    document = Document(
        id=uuid4(),
        library_id=library.id,
        name="Doc",
        created_at=datetime.utcnow(),
    )
    await doc_repo.create(document)

    # Create chunks
    num_chunks = 5
    vectors = []
    chunk_ids = []

    for i in range(num_chunks):
        embedding = np.random.rand(128).tolist()
        chunk = Chunk(
            id=uuid4(),
            document_id=document.id,
            text=f"Chunk {i}",
            embedding=embedding,
            metadata={"index": i},
            created_at=datetime.utcnow(),
        )
        chunk_ids.append(chunk.id)
        vectors.append(embedding)
        await chunk_repo.create(chunk, vector_index=i)

    # Save vectors
    vectors_array = np.array(vectors, dtype=np.float32)
    vector_storage.save(str(library.id), vectors_array)

    # Simulate: index file is missing/corrupted
    # Load chunks and rebuild index
    chunks_with_indices = await chunk_repo.list_by_library(str(library.id))
    loaded_vectors = vector_storage.load(str(library.id))

    # Rebuild index
    rebuilt_index = FlatIndex(distance_metric=DistanceMetric.COSINE)
    for chunk, vector_index in chunks_with_indices:
        # Attach embedding from vectors array
        chunk_with_embedding = Chunk(
            id=chunk.id,
            document_id=chunk.document_id,
            text=chunk.text,
            embedding=loaded_vectors[vector_index].tolist(),
            metadata=chunk.metadata,
            created_at=chunk.created_at,
        )
        rebuilt_index.add(chunk_with_embedding)

    # Verify rebuilt index works
    query = np.random.rand(128).tolist()
    results = rebuilt_index.search(query_embedding=query, k=5)
    assert len(results) == 5

    # Verify all original chunks are in the index
    found_ids = {chunk.id for chunk, _ in results}
    assert found_ids.issubset(set(chunk_ids))
