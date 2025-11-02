"""End-to-end tests for persistence across app restarts."""

import pytest
import pytest_asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from uuid import uuid4
import numpy as np

from vector_db.core.persistence.database import init_database, close_database
from vector_db.core.settings import Settings
from vector_db.repositories.persistent_library import PersistentLibraryRepository
from vector_db.models import Library, Chunk, IndexConfig


@pytest_asyncio.fixture
async def temp_data_dir():
    """Create a temporary data directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_persistence_across_restart(temp_data_dir):
    """Test that data persists across app restart."""

    # Create settings for temp directory
    settings = Settings()
    settings.DATA_DIR = temp_data_dir

    # First session: Create data
    library_id = uuid4()
    library_name = "Test Persistence Library"
    chunk_text = "This is a test chunk"

    # Initialize database and repository
    await init_database(settings.DATABASE_PATH)

    lib_repo = PersistentLibraryRepository(settings)
    await lib_repo.initialize()

    # Create library
    library = Library(
        id=library_id,
        name=library_name,
        index_config=IndexConfig(
            index_type="flat",
            distance_metric="cosine",
        ),
        created_at=datetime.utcnow(),
    )
    created_library = await lib_repo.create(library)

    # Add a chunk to the index
    chunk = Chunk(
        id=uuid4(),
        document_id=uuid4(),
        text=chunk_text,
        embedding=[0.1] * 128,
        metadata={"test": True},
        created_at=datetime.utcnow(),
    )
    await lib_repo.add_chunk_to_index(library_id, chunk)

    # Verify files exist
    assert settings.DATABASE_PATH.exists()
    assert (settings.VECTORS_DIR / f"{library_id}.npy").exists()
    assert (settings.INDEXES_DIR / f"{library_id}.pkl").exists()

    # Close database
    await close_database()

    # Second session: Restart and verify data loaded
    await init_database(settings.DATABASE_PATH)

    lib_repo = PersistentLibraryRepository(settings)
    await lib_repo.initialize()

    # Verify library exists in memory
    loaded_library = lib_repo.get(library_id)
    assert loaded_library is not None
    assert loaded_library.name == library_name
    assert loaded_library.id == library_id

    # Verify index was loaded
    index = lib_repo.get_index(library_id)
    assert index is not None
    assert index.size() == 1

    # Verify we can search
    from vector_db.models import VectorQuery
    query = VectorQuery(
        embedding=[0.1] * 128,
        k=5,
    )
    results = lib_repo.search(library_id, query)
    assert results.total_results == 1
    assert results.results[0].chunk.text == chunk_text

    # Cleanup
    await close_database()


@pytest.mark.asyncio
async def test_multiple_libraries_persistence(temp_data_dir):
    """Test that multiple libraries persist correctly."""

    # Create settings
    settings = Settings()
    settings.DATA_DIR = temp_data_dir

    # Initialize
    await init_database(settings.DATABASE_PATH)

    lib_repo = PersistentLibraryRepository(settings)
    await lib_repo.initialize()

    # Create 3 libraries with different configurations
    libraries = []
    for i in range(3):
        lib_id = uuid4()
        library = Library(
            id=lib_id,
            name=f"Library {i}",
            index_config=IndexConfig(
                index_type="flat",
                distance_metric="cosine",
            ),
            created_at=datetime.utcnow(),
        )
        await lib_repo.create(library)
        libraries.append(library)

        # Add chunks to each
        for j in range(5):
            chunk = Chunk(
                id=uuid4(),
                document_id=uuid4(),
                text=f"Library {i} Chunk {j}",
                embedding=np.random.rand(128).tolist(),
                metadata={"lib": i, "chunk": j},
                created_at=datetime.utcnow(),
            )
            await lib_repo.add_chunk_to_index(lib_id, chunk)

    await close_database()

    # Restart
    await init_database(settings.DATABASE_PATH)

    lib_repo = PersistentLibraryRepository(settings)
    await lib_repo.initialize()

    # Verify all libraries loaded
    all_libs = lib_repo.get_all()
    assert len(all_libs) == 3

    # Verify each library has correct number of chunks
    for library in libraries:
        index = lib_repo.get_index(library.id)
        assert index is not None
        assert index.size() == 5

    await close_database()


@pytest.mark.asyncio
async def test_delete_persistence(temp_data_dir):
    """Test that deletions persist correctly."""

    settings = Settings()
    settings.DATA_DIR = temp_data_dir

    # Initialize
    await init_database(settings.DATABASE_PATH)

    lib_repo = PersistentLibraryRepository(settings)
    await lib_repo.initialize()

    # Create library
    library_id = uuid4()
    library = Library(
        id=library_id,
        name="To Be Deleted",
        index_config=IndexConfig(index_type="flat", distance_metric="cosine"),
        created_at=datetime.utcnow(),
    )
    await lib_repo.create(library)

    # Add chunk
    chunk = Chunk(
        id=uuid4(),
        document_id=uuid4(),
        text="Test",
        embedding=[0.1] * 128,
        metadata={},
        created_at=datetime.utcnow(),
    )
    await lib_repo.add_chunk_to_index(library_id, chunk)

    # Verify files exist
    assert (settings.VECTORS_DIR / f"{library_id}.npy").exists()
    assert (settings.INDEXES_DIR / f"{library_id}.pkl").exists()

    # Delete library
    await lib_repo.delete(library_id)

    # Verify files deleted
    assert not (settings.VECTORS_DIR / f"{library_id}.npy").exists()
    assert not (settings.INDEXES_DIR / f"{library_id}.pkl").exists()

    await close_database()

    # Restart and verify deletion persisted
    await init_database(settings.DATABASE_PATH)

    lib_repo = PersistentLibraryRepository(settings)
    await lib_repo.initialize()

    # Verify library not in memory
    all_libs = lib_repo.get_all()
    assert len(all_libs) == 0

    await close_database()
