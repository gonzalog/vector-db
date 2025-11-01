"""SQLAlchemy database models."""

from datetime import datetime
from sqlalchemy import Column, String, Text, Integer, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class LibraryModel(Base):
    """SQLAlchemy model for libraries table."""

    __tablename__ = "libraries"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    index_config = Column(Text, nullable=False)  # JSON string
    meta_data = Column("metadata", Text, nullable=True)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    documents = relationship("DocumentModel", back_populates="library", cascade="all, delete-orphan")


class DocumentModel(Base):
    """SQLAlchemy model for documents table."""

    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    library_id = Column(String, ForeignKey("libraries.id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=False)
    meta_data = Column("metadata", Text, nullable=True)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    library = relationship("LibraryModel", back_populates="documents")
    chunks = relationship("ChunkModel", back_populates="document", cascade="all, delete-orphan")


class ChunkModel(Base):
    """SQLAlchemy model for chunks table."""

    __tablename__ = "chunks"

    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    text = Column(Text, nullable=False)
    meta_data = Column("metadata", Text, nullable=True)  # JSON string
    vector_index = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    document = relationship("DocumentModel", back_populates="chunks")
