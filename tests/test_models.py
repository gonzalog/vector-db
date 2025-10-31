"""Tests for Pydantic models."""

from uuid import UUID, uuid4

import pytest

from vector_db.models import (
    Chunk,
    ChunkCreate,
    ChunkMetadata,
    Document,
    DocumentCreate,
    DocumentMetadata,
    Library,
    LibraryCreate,
    LibraryMetadata,
    SearchResult,
    VectorQuery,
)


class TestChunkModels:
    """Tests for Chunk models."""

    def test_chunk_creation(self):
        """Test creating a chunk with all fields."""
        doc_id = uuid4()
        metadata = ChunkMetadata(
            source="test.pdf", page_number=1, position=0, tags=["test"]
        )
        chunk = Chunk(
            document_id=doc_id,
            text="Sample text",
            embedding=[0.1, 0.2, 0.3],
            metadata=metadata,
        )

        assert isinstance(chunk.id, UUID)
        assert chunk.document_id == doc_id
        assert chunk.text == "Sample text"
        assert chunk.embedding == [0.1, 0.2, 0.3]
        assert chunk.metadata.source == "test.pdf"
        assert chunk.metadata.page_number == 1

    def test_chunk_create_schema(self):
        """Test ChunkCreate schema validation."""
        chunk_create = ChunkCreate(
            text="Sample text",
            embedding=[0.1, 0.2, 0.3],
        )

        assert chunk_create.text == "Sample text"
        assert chunk_create.embedding == [0.1, 0.2, 0.3]

    def test_chunk_text_validation(self):
        """Test that empty text is not allowed."""
        with pytest.raises(ValueError):
            ChunkCreate(text="", embedding=[0.1, 0.2, 0.3])


class TestDocumentModels:
    """Tests for Document models."""

    def test_document_creation(self):
        """Test creating a document with all fields."""
        lib_id = uuid4()
        metadata = DocumentMetadata(
            title="Test Document",
            author="John Doe",
            document_type="pdf",
            tags=["test"],
        )
        document = Document(
            library_id=lib_id, name="test_doc", metadata=metadata
        )

        assert isinstance(document.id, UUID)
        assert document.library_id == lib_id
        assert document.name == "test_doc"
        assert document.metadata.title == "Test Document"
        assert document.metadata.author == "John Doe"

    def test_document_create_schema(self):
        """Test DocumentCreate schema validation."""
        doc_create = DocumentCreate(
            name="test_doc",
            metadata=DocumentMetadata(title="Test"),
        )

        assert doc_create.name == "test_doc"
        assert doc_create.metadata.title == "Test"

    def test_document_name_validation(self):
        """Test that empty name is not allowed."""
        with pytest.raises(ValueError):
            DocumentCreate(name="")


class TestLibraryModels:
    """Tests for Library models."""

    def test_library_creation(self):
        """Test creating a library with all fields."""
        metadata = LibraryMetadata(
            description="Test library",
            owner="test@example.com",
            tags=["test"],
            embedding_model="cohere-embed-english-v3.0",
            embedding_dimension=1024,
        )
        library = Library(name="test_library", metadata=metadata)

        assert isinstance(library.id, UUID)
        assert library.name == "test_library"
        assert library.metadata.description == "Test library"
        assert library.metadata.embedding_dimension == 1024

    def test_library_create_schema(self):
        """Test LibraryCreate schema validation."""
        lib_create = LibraryCreate(
            name="test_library",
            metadata=LibraryMetadata(description="Test"),
        )

        assert lib_create.name == "test_library"
        assert lib_create.metadata.description == "Test"

    def test_library_name_validation(self):
        """Test that empty name is not allowed."""
        with pytest.raises(ValueError):
            LibraryCreate(name="")


class TestQueryModels:
    """Tests for Query models."""

    def test_vector_query_creation(self):
        """Test creating a vector query."""
        query = VectorQuery(embedding=[0.1, 0.2, 0.3], k=5)

        assert query.embedding == [0.1, 0.2, 0.3]
        assert query.k == 5
        assert query.metadata_filter is None

    def test_vector_query_with_filters(self):
        """Test creating a vector query with metadata filters."""
        query = VectorQuery(
            embedding=[0.1, 0.2, 0.3],
            k=10,
            metadata_filter={"author": "John Doe", "tags": ["important"]},
        )

        assert query.k == 10
        assert query.metadata_filter["author"] == "John Doe"

    def test_vector_query_k_validation(self):
        """Test k parameter validation."""
        with pytest.raises(ValueError):
            VectorQuery(embedding=[0.1, 0.2, 0.3], k=0)

        with pytest.raises(ValueError):
            VectorQuery(embedding=[0.1, 0.2, 0.3], k=1001)

    def test_search_result_creation(self):
        """Test creating a search result."""
        chunk = Chunk(
            document_id=uuid4(),
            text="Sample text",
            embedding=[0.1, 0.2, 0.3],
        )
        result = SearchResult(chunk=chunk, score=0.95, distance=0.05)

        assert result.chunk.text == "Sample text"
        assert result.score == 0.95
        assert result.distance == 0.05
