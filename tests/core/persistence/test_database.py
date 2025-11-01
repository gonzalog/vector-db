"""Tests for database module."""

import pytest
import pytest_asyncio
from pathlib import Path
import tempfile
import shutil
from sqlalchemy import text

from vector_db.core.persistence.database import Database, init_database, close_database


@pytest_asyncio.fixture
async def temp_db_path():
    """Create a temporary database path."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test.db"
    yield db_path
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_database_initialization(temp_db_path):
    """Test database initialization creates schema."""
    db = Database(temp_db_path)
    await db.connect()

    # Check that tables were created using SQLAlchemy
    async with db.get_session() as session:
        result = await session.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        )
        tables = [row[0] for row in result.fetchall()]

    assert "libraries" in tables
    assert "documents" in tables
    assert "chunks" in tables

    await db.disconnect()


@pytest.mark.asyncio
async def test_database_foreign_keys_enabled(temp_db_path):
    """Test that foreign keys are enabled."""
    db = Database(temp_db_path)
    await db.connect()

    async with db.get_session() as session:
        result = await session.execute(text("PRAGMA foreign_keys"))
        row = result.fetchone()
        assert row[0] == 1  # Foreign keys enabled

    await db.disconnect()


@pytest.mark.asyncio
async def test_init_database_global():
    """Test global database initialization."""
    temp_dir = tempfile.mkdtemp()
    try:
        db_path = Path(temp_dir) / "test.db"

        db = await init_database(db_path)
        assert db is not None
        assert db.session_factory is not None

        await close_database()
    finally:
        shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_database_creates_parent_directory(temp_db_path):
    """Test that database creates parent directory if it doesn't exist."""
    # Make sure parent doesn't exist
    if temp_db_path.parent.exists():
        shutil.rmtree(temp_db_path.parent)

    db = Database(temp_db_path)
    await db.connect()

    assert temp_db_path.exists()
    assert temp_db_path.parent.exists()

    await db.disconnect()
