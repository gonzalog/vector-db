"""Tests for database repositories."""

import pytest
import pytest_asyncio
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
from uuid import uuid4

from vector_db.core.persistence.database import Database
from vector_db.core.persistence.repositories import (
    LibraryRepository,
    DocumentRepository,
    ChunkRepository,
)
from vector_db.models import Library, Document, Chunk, IndexConfig


@pytest_asyncio.fixture
async def temp_db():
    """Create a temporary database."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test.db"
    db = Database(db_path)
    await db.connect()
    yield db
    await db.disconnect()
    shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_library_create_and_get(temp_db):
    """Test creating and retrieving a library."""
    repo = LibraryRepository(temp_db)

    library = Library(
        id=uuid4(),
        name="Test Library",
        index_config=IndexConfig(
            index_type="flat",
            distance_metric="cosine",
        ),
        metadata={"description": "Test"},
        created_at=datetime.utcnow(),
    )

    await repo.create(library)

    # Retrieve
    retrieved = await repo.get(str(library.id))
    assert retrieved is not None
    assert retrieved.id == library.id
    assert retrieved.name == library.name
    assert retrieved.index_config.index_type == "flat"


@pytest.mark.asyncio
async def test_library_list(temp_db):
    """Test listing libraries."""
    repo = LibraryRepository(temp_db)

    # Create multiple libraries
    for i in range(5):
        library = Library(
            id=uuid4(),
            name=f"Library {i}",
            index_config=IndexConfig(index_type="flat", distance_metric="cosine"),
            created_at=datetime.utcnow(),
        )
        await repo.create(library)

    # List all
    libraries = await repo.list(skip=0, limit=10)
    assert len(libraries) == 5


@pytest.mark.asyncio
async def test_library_list_pagination(temp_db):
    """Test library list pagination."""
    repo = LibraryRepository(temp_db)

    # Create libraries
    for i in range(10):
        library = Library(
            id=uuid4(),
            name=f"Library {i}",
            index_config=IndexConfig(index_type="flat", distance_metric="cosine"),
            created_at=datetime.utcnow(),
        )
        await repo.create(library)

    # Test pagination
    first_page = await repo.list(skip=0, limit=5)
    assert len(first_page) == 5

    second_page = await repo.list(skip=5, limit=5)
    assert len(second_page) == 5

    # Ensure different results
    first_ids = {lib.id for lib in first_page}
    second_ids = {lib.id for lib in second_page}
    assert len(first_ids & second_ids) == 0


@pytest.mark.asyncio
async def test_library_delete(temp_db):
    """Test deleting a library."""
    repo = LibraryRepository(temp_db)

    library = Library(
        id=uuid4(),
        name="To Delete",
        index_config=IndexConfig(index_type="flat", distance_metric="cosine"),
        created_at=datetime.utcnow(),
    )

    await repo.create(library)
    await repo.delete(str(library.id))

    # Should not be found
    result = await repo.get(str(library.id))
    assert result is None


@pytest.mark.asyncio
async def test_document_create_and_get(temp_db):
    """Test creating and retrieving a document."""
    lib_repo = LibraryRepository(temp_db)
    doc_repo = DocumentRepository(temp_db)

    # Create library first
    library = Library(
        id=uuid4(),
        name="Test Library",
        index_config=IndexConfig(index_type="flat", distance_metric="cosine"),
        created_at=datetime.utcnow(),
    )
    await lib_repo.create(library)

    # Create document
    document = Document(
        id=uuid4(),
        library_id=library.id,
        name="Test Document",
        metadata={"author": "Test"},
        created_at=datetime.utcnow(),
    )
    await doc_repo.create(document)

    # Retrieve
    retrieved = await doc_repo.get(str(document.id))
    assert retrieved is not None
    assert retrieved.id == document.id
    assert retrieved.name == document.name


@pytest.mark.asyncio
async def test_document_cascade_delete(temp_db):
    """Test that deleting library cascades to documents."""
    lib_repo = LibraryRepository(temp_db)
    doc_repo = DocumentRepository(temp_db)

    # Create library and document
    library = Library(
        id=uuid4(),
        name="Test Library",
        index_config=IndexConfig(index_type="flat", distance_metric="cosine"),
        created_at=datetime.utcnow(),
    )
    await lib_repo.create(library)

    document = Document(
        id=uuid4(),
        library_id=library.id,
        name="Test Document",
        created_at=datetime.utcnow(),
    )
    await doc_repo.create(document)

    # Delete library
    await lib_repo.delete(str(library.id))

    # Document should also be deleted
    result = await doc_repo.get(str(document.id))
    assert result is None


@pytest.mark.asyncio
async def test_chunk_create_and_get(temp_db):
    """Test creating and retrieving a chunk."""
    lib_repo = LibraryRepository(temp_db)
    doc_repo = DocumentRepository(temp_db)
    chunk_repo = ChunkRepository(temp_db)

    # Create library and document
    library = Library(
        id=uuid4(),
        name="Test Library",
        index_config=IndexConfig(index_type="flat", distance_metric="cosine"),
        created_at=datetime.utcnow(),
    )
    await lib_repo.create(library)

    document = Document(
        id=uuid4(),
        library_id=library.id,
        name="Test Document",
        created_at=datetime.utcnow(),
    )
    await doc_repo.create(document)

    # Create chunk
    chunk = Chunk(
        id=uuid4(),
        document_id=document.id,
        text="Test chunk text",
        embedding=[0.1] * 128,  # Dummy embedding
        metadata={"index": 0},
        created_at=datetime.utcnow(),
    )
    vector_index = 0
    await chunk_repo.create(chunk, vector_index)

    # Retrieve
    result = await chunk_repo.get(str(chunk.id))
    assert result is not None
    retrieved_chunk, retrieved_index = result
    assert retrieved_chunk.id == chunk.id
    assert retrieved_chunk.text == chunk.text
    assert retrieved_index == vector_index


@pytest.mark.asyncio
async def test_chunk_list_by_library(temp_db):
    """Test listing chunks by library."""
    lib_repo = LibraryRepository(temp_db)
    doc_repo = DocumentRepository(temp_db)
    chunk_repo = ChunkRepository(temp_db)

    # Create library and document
    library = Library(
        id=uuid4(),
        name="Test Library",
        index_config=IndexConfig(index_type="flat", distance_metric="cosine"),
        created_at=datetime.utcnow(),
    )
    await lib_repo.create(library)

    document = Document(
        id=uuid4(),
        library_id=library.id,
        name="Test Document",
        created_at=datetime.utcnow(),
    )
    await doc_repo.create(document)

    # Create chunks
    for i in range(5):
        chunk = Chunk(
            id=uuid4(),
            document_id=document.id,
            text=f"Chunk {i}",
            embedding=[float(i)] * 128,  # Dummy embedding
            metadata={"index": i},
            created_at=datetime.utcnow(),
        )
        await chunk_repo.create(chunk, vector_index=i)

    # List chunks
    chunks_with_indices = await chunk_repo.list_by_library(str(library.id))
    assert len(chunks_with_indices) == 5

    # Verify they're sorted by vector_index
    for i, (chunk, vector_index) in enumerate(chunks_with_indices):
        assert vector_index == i
