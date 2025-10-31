"""Shared test fixtures for all tests."""

import pytest
import pytest_asyncio
import tempfile
import shutil
from pathlib import Path
from fastapi.testclient import TestClient

from vector_db.main import create_app
from vector_db.core.persistence.database import init_database, close_database
from vector_db.core.settings import Settings
from vector_db.repositories.registry import initialize_repositories


@pytest_asyncio.fixture
async def temp_data_dir():
    """Create a temporary data directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest_asyncio.fixture
async def app_with_persistence(temp_data_dir):
    """Create app with persistence initialized."""
    settings = Settings()
    settings.DATA_DIR = temp_data_dir

    # Initialize database and repositories
    await init_database(settings.DATABASE_PATH)
    await initialize_repositories()

    app = create_app()

    yield app

    # Cleanup
    await close_database()


@pytest.fixture
def client(app_with_persistence):
    """Create test client."""
    return TestClient(app_with_persistence, raise_server_exceptions=True)
