"""SQLAlchemy database connection and schema management."""

from pathlib import Path
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import event
from sqlalchemy.pool import StaticPool

from vector_db.core.persistence.models import Base


class Database:
    """Async SQLAlchemy database manager."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None

    async def connect(self) -> None:
        """Connect to database and initialize schema."""
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create async engine for SQLite
        # Using StaticPool for SQLite to avoid connection issues
        database_url = f"sqlite+aiosqlite:///{self.db_path}"
        self._engine = create_async_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False,
        )

        # Enable foreign keys for SQLite
        @event.listens_for(self._engine.sync_engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        # Create session factory
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # Initialize schema
        await self._init_schema()

    async def disconnect(self) -> None:
        """Close database connection."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None

    async def _init_schema(self) -> None:
        """Initialize database schema using SQLAlchemy models."""
        from sqlalchemy import text

        async with self._engine.begin() as conn:
            # Create all tables defined in Base metadata
            await conn.run_sync(Base.metadata.create_all)

            # Create indexes (SQLite only allows one statement at a time)
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_documents_library_id ON documents(library_id)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chunks_vector_index ON chunks(vector_index)"))

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Get session factory."""
        if not self._session_factory:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._session_factory

    def get_session(self) -> AsyncSession:
        """Get a new database session."""
        return self.session_factory()


# Global database instance
_db: Optional[Database] = None


def get_database() -> Database:
    """Get global database instance."""
    if not _db:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _db


async def init_database(db_path: Path) -> Database:
    """Initialize global database instance."""
    global _db
    _db = Database(db_path)
    await _db.connect()
    return _db


async def close_database() -> None:
    """Close global database connection."""
    global _db
    if _db:
        await _db.disconnect()
        _db = None
