"""Repository layer for database operations using SQLAlchemy ORM."""

import json
from typing import Optional
from datetime import datetime
from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from vector_db.core.persistence.database import Database
from vector_db.core.persistence.models import LibraryModel, DocumentModel, ChunkModel
from vector_db.models import Library, Document, Chunk


class LibraryRepository:
    """Repository for library operations using SQLAlchemy."""

    def __init__(self, db: Database):
        self.db = db

    async def create(self, library: Library) -> None:
        """Create a new library."""
        async with self.db.get_session() as session:
            # Convert Pydantic models to dicts if needed
            metadata_dict = None
            if library.metadata:
                metadata_dict = (
                    library.metadata.model_dump()
                    if hasattr(library.metadata, "model_dump")
                    else library.metadata
                )

            library_model = LibraryModel(
                id=str(library.id),
                name=library.name,
                index_config=json.dumps(library.index_config.model_dump()),
                meta_data=json.dumps(metadata_dict) if metadata_dict else None,
                created_at=library.created_at,
            )
            session.add(library_model)
            await session.commit()

    async def get(self, library_id: str) -> Optional[Library]:
        """Get a library by ID."""
        async with self.db.get_session() as session:
            stmt = select(LibraryModel).where(LibraryModel.id == library_id)
            result = await session.execute(stmt)
            library_model = result.scalar_one_or_none()

            if not library_model:
                return None

            return Library(
                id=library_model.id,
                name=library_model.name,
                index_config=json.loads(library_model.index_config),
                metadata=json.loads(library_model.meta_data) if library_model.meta_data else None,
                created_at=library_model.created_at,
            )

    async def list(self, skip: int = 0, limit: int = 100) -> list[Library]:
        """List all libraries with pagination."""
        async with self.db.get_session() as session:
            stmt = (
                select(LibraryModel)
                .order_by(LibraryModel.created_at.desc())
                .offset(skip)
                .limit(limit)
            )
            result = await session.execute(stmt)
            library_models = result.scalars().all()

            return [
                Library(
                    id=model.id,
                    name=model.name,
                    index_config=json.loads(model.index_config),
                    metadata=json.loads(model.meta_data) if model.meta_data else None,
                    created_at=model.created_at,
                )
                for model in library_models
            ]

    async def update(self, library_id: str, library: Library) -> None:
        """Update a library."""
        async with self.db.get_session() as session:
            metadata_dict = None
            if library.metadata:
                metadata_dict = (
                    library.metadata.model_dump()
                    if hasattr(library.metadata, "model_dump")
                    else library.metadata
                )

            stmt = (
                update(LibraryModel)
                .where(LibraryModel.id == library_id)
                .values(
                    name=library.name,
                    meta_data=json.dumps(metadata_dict) if metadata_dict else None,
                    index_config=json.dumps(library.index_config.model_dump()),
                )
            )
            await session.execute(stmt)
            await session.commit()

    async def delete(self, library_id: str) -> None:
        """Delete a library (cascades to documents and chunks)."""
        async with self.db.get_session() as session:
            stmt = delete(LibraryModel).where(LibraryModel.id == library_id)
            await session.execute(stmt)
            await session.commit()


class DocumentRepository:
    """Repository for document operations using SQLAlchemy."""

    def __init__(self, db: Database):
        self.db = db

    async def create(self, document: Document) -> None:
        """Create a new document."""
        async with self.db.get_session() as session:
            metadata_dict = None
            if document.metadata:
                metadata_dict = (
                    document.metadata.model_dump()
                    if hasattr(document.metadata, "model_dump")
                    else document.metadata
                )

            document_model = DocumentModel(
                id=str(document.id),
                library_id=str(document.library_id),
                name=document.name,
                meta_data=json.dumps(metadata_dict) if metadata_dict else None,
                created_at=document.created_at,
            )
            session.add(document_model)
            await session.commit()

    async def get(self, document_id: str) -> Optional[Document]:
        """Get a document by ID."""
        async with self.db.get_session() as session:
            stmt = select(DocumentModel).where(DocumentModel.id == document_id)
            result = await session.execute(stmt)
            document_model = result.scalar_one_or_none()

            if not document_model:
                return None

            return Document(
                id=document_model.id,
                library_id=document_model.library_id,
                name=document_model.name,
                metadata=json.loads(document_model.meta_data) if document_model.meta_data else None,
                created_at=document_model.created_at,
            )

    async def list_by_library(
        self, library_id: str, skip: int = 0, limit: int = 100
    ) -> list[Document]:
        """List documents in a library with pagination."""
        async with self.db.get_session() as session:
            stmt = (
                select(DocumentModel)
                .where(DocumentModel.library_id == library_id)
                .order_by(DocumentModel.created_at.desc())
                .offset(skip)
                .limit(limit)
            )
            result = await session.execute(stmt)
            document_models = result.scalars().all()

            return [
                Document(
                    id=model.id,
                    library_id=model.library_id,
                    name=model.name,
                    metadata=json.loads(model.meta_data) if model.meta_data else None,
                    created_at=model.created_at,
                )
                for model in document_models
            ]

    async def update(self, document_id: str, document: Document) -> None:
        """Update a document."""
        async with self.db.get_session() as session:
            metadata_dict = None
            if document.metadata:
                metadata_dict = (
                    document.metadata.model_dump()
                    if hasattr(document.metadata, "model_dump")
                    else document.metadata
                )

            stmt = (
                update(DocumentModel)
                .where(DocumentModel.id == document_id)
                .values(
                    name=document.name,
                    meta_data=json.dumps(metadata_dict) if metadata_dict else None,
                )
            )
            await session.execute(stmt)
            await session.commit()

    async def count_by_library(self, library_id: str) -> int:
        """Count documents in a library."""
        async with self.db.get_session() as session:
            stmt = (
                select(func.count())
                .select_from(DocumentModel)
                .where(DocumentModel.library_id == library_id)
            )
            result = await session.execute(stmt)
            return result.scalar_one()

    async def delete(self, document_id: str) -> None:
        """Delete a document (cascades to chunks)."""
        async with self.db.get_session() as session:
            stmt = delete(DocumentModel).where(DocumentModel.id == document_id)
            await session.execute(stmt)
            await session.commit()


class ChunkRepository:
    """Repository for chunk operations using SQLAlchemy."""

    def __init__(self, db: Database):
        self.db = db

    async def create(self, chunk: Chunk, vector_index: int) -> None:
        """Create a new chunk."""
        async with self.db.get_session() as session:
            metadata_dict = None
            if chunk.metadata:
                metadata_dict = (
                    chunk.metadata.model_dump()
                    if hasattr(chunk.metadata, "model_dump")
                    else chunk.metadata
                )

            chunk_model = ChunkModel(
                id=str(chunk.id),
                document_id=str(chunk.document_id),
                text=chunk.text,
                meta_data=json.dumps(metadata_dict) if metadata_dict else None,
                vector_index=vector_index,
                created_at=chunk.created_at,
            )
            session.add(chunk_model)
            await session.commit()

    async def get(self, chunk_id: str) -> Optional[tuple[Chunk, int]]:
        """Get a chunk by ID, returns (chunk, vector_index)."""
        async with self.db.get_session() as session:
            stmt = select(ChunkModel).where(ChunkModel.id == chunk_id)
            result = await session.execute(stmt)
            chunk_model = result.scalar_one_or_none()

            if not chunk_model:
                return None

            chunk = Chunk(
                id=chunk_model.id,
                document_id=chunk_model.document_id,
                text=chunk_model.text,
                embedding=[],  # Placeholder - actual embedding is in vector storage
                metadata=json.loads(chunk_model.meta_data) if chunk_model.meta_data else None,
                created_at=chunk_model.created_at,
            )
            return chunk, chunk_model.vector_index

    async def list_by_document(self, document_id: str) -> list[tuple[Chunk, int]]:
        """List chunks in a document, returns list of (chunk, vector_index)."""
        async with self.db.get_session() as session:
            stmt = (
                select(ChunkModel)
                .where(ChunkModel.document_id == document_id)
                .order_by(ChunkModel.created_at.asc())
            )
            result = await session.execute(stmt)
            chunk_models = result.scalars().all()

            return [
                (
                    Chunk(
                        id=model.id,
                        document_id=model.document_id,
                        text=model.text,
                        embedding=[],  # Placeholder
                        metadata=json.loads(model.meta_data) if model.meta_data else None,
                        created_at=model.created_at,
                    ),
                    model.vector_index,
                )
                for model in chunk_models
            ]

    async def list_by_library(self, library_id: str) -> list[tuple[Chunk, int]]:
        """List all chunks in a library, returns list of (chunk, vector_index)."""
        async with self.db.get_session() as session:
            stmt = (
                select(ChunkModel)
                .join(DocumentModel)
                .where(DocumentModel.library_id == library_id)
                .order_by(ChunkModel.vector_index.asc())
            )
            result = await session.execute(stmt)
            chunk_models = result.scalars().all()

            return [
                (
                    Chunk(
                        id=model.id,
                        document_id=model.document_id,
                        text=model.text,
                        embedding=[],  # Placeholder
                        metadata=json.loads(model.meta_data) if model.meta_data else None,
                        created_at=model.created_at,
                    ),
                    model.vector_index,
                )
                for model in chunk_models
            ]

    async def update(self, chunk_id: str, chunk: Chunk) -> None:
        """Update a chunk."""
        async with self.db.get_session() as session:
            metadata_dict = None
            if chunk.metadata:
                metadata_dict = (
                    chunk.metadata.model_dump()
                    if hasattr(chunk.metadata, "model_dump")
                    else chunk.metadata
                )

            stmt = (
                update(ChunkModel)
                .where(ChunkModel.id == chunk_id)
                .values(
                    text=chunk.text,
                    meta_data=json.dumps(metadata_dict) if metadata_dict else None,
                )
            )
            await session.execute(stmt)
            await session.commit()

    async def count_by_document(self, document_id: str) -> int:
        """Count chunks in a document."""
        async with self.db.get_session() as session:
            stmt = (
                select(func.count())
                .select_from(ChunkModel)
                .where(ChunkModel.document_id == document_id)
            )
            result = await session.execute(stmt)
            return result.scalar_one()

    async def update_vector_index(self, chunk_id: str, vector_index: int) -> None:
        """Update the vector_index for a chunk."""
        async with self.db.get_session() as session:
            stmt = (
                update(ChunkModel)
                .where(ChunkModel.id == chunk_id)
                .values(vector_index=vector_index)
            )
            await session.execute(stmt)
            await session.commit()

    async def delete(self, chunk_id: str) -> None:
        """Delete a chunk."""
        async with self.db.get_session() as session:
            stmt = delete(ChunkModel).where(ChunkModel.id == chunk_id)
            await session.execute(stmt)
            await session.commit()
