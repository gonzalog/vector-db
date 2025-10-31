"""SQLite database connection and schema management."""

import aiosqlite
from pathlib import Path
from typing import Optional


class Database:
    """Async SQLite database manager."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        """Connect to database and initialize schema."""
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect to database
        self._conn = await aiosqlite.connect(str(self.db_path))
        self._conn.row_factory = aiosqlite.Row

        # Enable foreign keys
        await self._conn.execute("PRAGMA foreign_keys = ON")

        # Initialize schema
        await self._init_schema()

    async def disconnect(self) -> None:
        """Close database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def _init_schema(self) -> None:
        """Initialize database schema."""
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS libraries (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                index_config TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                library_id TEXT NOT NULL,
                name TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (library_id) REFERENCES libraries(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata TEXT,
                vector_index INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            );

            -- Indexes for better query performance
            CREATE INDEX IF NOT EXISTS idx_documents_library_id ON documents(library_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_vector_index ON chunks(vector_index);
        """)
        await self._conn.commit()

    @property
    def conn(self) -> aiosqlite.Connection:
        """Get database connection."""
        if not self._conn:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._conn


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
