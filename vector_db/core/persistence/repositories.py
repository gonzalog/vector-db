"""Repository layer for database operations."""

import json
from typing import Optional
from datetime import datetime

from vector_db.core.persistence.database import Database
from vector_db.models import Library, Document, Chunk


class LibraryRepository:
    """Repository for library operations."""

    def __init__(self, db: Database):
        self.db = db

    async def create(self, library: Library) -> None:
        """Create a new library."""
        await self.db.conn.execute(
            """
            INSERT INTO libraries (id, name, index_config, metadata, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                library.id,
                library.name,
                json.dumps(library.index_config),
                json.dumps(library.metadata) if library.metadata else None,
                library.created_at.isoformat(),
            ),
        )
        await self.db.conn.commit()

    async def get(self, library_id: str) -> Optional[Library]:
        """Get a library by ID."""
        async with self.db.conn.execute(
            "SELECT * FROM libraries WHERE id = ?",
            (library_id,),
        ) as cursor:
            row = await cursor.fetchone()

        if not row:
            return None

        return Library(
            id=row["id"],
            name=row["name"],
            index_config=json.loads(row["index_config"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    async def list(self, skip: int = 0, limit: int = 100) -> list[Library]:
        """List all libraries with pagination."""
        async with self.db.conn.execute(
            "SELECT * FROM libraries ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, skip),
        ) as cursor:
            rows = await cursor.fetchall()

        return [
            Library(
                id=row["id"],
                name=row["name"],
                index_config=json.loads(row["index_config"]),
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    async def delete(self, library_id: str) -> None:
        """Delete a library (cascades to documents and chunks)."""
        await self.db.conn.execute(
            "DELETE FROM libraries WHERE id = ?",
            (library_id,),
        )
        await self.db.conn.commit()


class DocumentRepository:
    """Repository for document operations."""

    def __init__(self, db: Database):
        self.db = db

    async def create(self, document: Document) -> None:
        """Create a new document."""
        await self.db.conn.execute(
            """
            INSERT INTO documents (id, library_id, name, metadata, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                document.id,
                document.library_id,
                document.name,
                json.dumps(document.metadata) if document.metadata else None,
                document.created_at.isoformat(),
            ),
        )
        await self.db.conn.commit()

    async def get(self, document_id: str) -> Optional[Document]:
        """Get a document by ID."""
        async with self.db.conn.execute(
            "SELECT * FROM documents WHERE id = ?",
            (document_id,),
        ) as cursor:
            row = await cursor.fetchone()

        if not row:
            return None

        return Document(
            id=row["id"],
            library_id=row["library_id"],
            name=row["name"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    async def list_by_library(
        self, library_id: str, skip: int = 0, limit: int = 100
    ) -> list[Document]:
        """List documents in a library with pagination."""
        async with self.db.conn.execute(
            """
            SELECT * FROM documents
            WHERE library_id = ?
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (library_id, limit, skip),
        ) as cursor:
            rows = await cursor.fetchall()

        return [
            Document(
                id=row["id"],
                library_id=row["library_id"],
                name=row["name"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    async def delete(self, document_id: str) -> None:
        """Delete a document (cascades to chunks)."""
        await self.db.conn.execute(
            "DELETE FROM documents WHERE id = ?",
            (document_id,),
        )
        await self.db.conn.commit()


class ChunkRepository:
    """Repository for chunk operations."""

    def __init__(self, db: Database):
        self.db = db

    async def create(self, chunk: Chunk, vector_index: int) -> None:
        """Create a new chunk."""
        await self.db.conn.execute(
            """
            INSERT INTO chunks (id, document_id, text, metadata, vector_index, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.id,
                chunk.document_id,
                chunk.text,
                json.dumps(chunk.metadata) if chunk.metadata else None,
                vector_index,
                chunk.created_at.isoformat(),
            ),
        )
        await self.db.conn.commit()

    async def get(self, chunk_id: str) -> Optional[tuple[Chunk, int]]:
        """Get a chunk by ID, returns (chunk, vector_index)."""
        async with self.db.conn.execute(
            "SELECT * FROM chunks WHERE id = ?",
            (chunk_id,),
        ) as cursor:
            row = await cursor.fetchone()

        if not row:
            return None

        chunk = Chunk(
            id=row["id"],
            document_id=row["document_id"],
            text=row["text"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
        )
        return chunk, row["vector_index"]

    async def list_by_document(self, document_id: str) -> list[tuple[Chunk, int]]:
        """List chunks in a document, returns list of (chunk, vector_index)."""
        async with self.db.conn.execute(
            """
            SELECT * FROM chunks
            WHERE document_id = ?
            ORDER BY created_at ASC
            """,
            (document_id,),
        ) as cursor:
            rows = await cursor.fetchall()

        return [
            (
                Chunk(
                    id=row["id"],
                    document_id=row["document_id"],
                    text=row["text"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else None,
                    created_at=datetime.fromisoformat(row["created_at"]),
                ),
                row["vector_index"],
            )
            for row in rows
        ]

    async def list_by_library(self, library_id: str) -> list[tuple[Chunk, int]]:
        """List all chunks in a library, returns list of (chunk, vector_index)."""
        async with self.db.conn.execute(
            """
            SELECT c.* FROM chunks c
            INNER JOIN documents d ON c.document_id = d.id
            WHERE d.library_id = ?
            ORDER BY c.vector_index ASC
            """,
            (library_id,),
        ) as cursor:
            rows = await cursor.fetchall()

        return [
            (
                Chunk(
                    id=row["id"],
                    document_id=row["document_id"],
                    text=row["text"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else None,
                    created_at=datetime.fromisoformat(row["created_at"]),
                ),
                row["vector_index"],
            )
            for row in rows
        ]

    async def delete(self, chunk_id: str) -> None:
        """Delete a chunk."""
        await self.db.conn.execute(
            "DELETE FROM chunks WHERE id = ?",
            (chunk_id,),
        )
        await self.db.conn.commit()
