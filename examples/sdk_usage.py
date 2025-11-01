"""Example usage of the Vector DB Python SDK.

This example demonstrates how to use the Vector DB SDK to:
1. Create a library
2. Create documents
3. Add chunks with embeddings using Cohere
4. Search for similar vectors with and without metadata filters
5. Delete a chunk and observe updated search results
6. Update and delete resources

Requirements:
- Set COHERE_API_KEY in your .env file
- Run the API server: uv run uvicorn vector_db.main:app --port 8000
"""

import os
from dotenv import load_dotenv
import cohere
from vector_db.sdk import VectorDBClient, NotFoundError, AlreadyExistsError

# Load environment variables from .env file
load_dotenv()


def get_cohere_client() -> cohere.Client:
    """Initialize Cohere client from environment variables."""
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError(
            "COHERE_API_KEY not found in environment variables. "
            "Please set it in your .env file."
        )
    return cohere.Client(api_key)


def generate_embeddings(texts: list[str], cohere_client: cohere.Client) -> list[list[float]]:
    """Generate embeddings for a list of texts using Cohere.

    Args:
        texts: List of text strings to embed
        cohere_client: Initialized Cohere client

    Returns:
        List of embedding vectors
    """
    response = cohere_client.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document",
    )
    return response.embeddings


def main():
    """Main example demonstrating SDK usage."""

    # Get configuration from environment
    api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")

    # Initialize the Vector DB client
    print("Connecting to Vector DB API...")
    print(f"  Base URL: {api_base_url}")
    client = VectorDBClient(base_url=api_base_url)

    # Initialize Cohere client
    print("Initializing Cohere client...")
    try:
        cohere_client = get_cohere_client()
        print("✓ Cohere client initialized")
    except ValueError as e:
        print(f"✗ {e}")
        return

    try:
        # 1. Create a library
        print("\n1. Creating a library...")
        library = client.create_library(
            name="SDK Example Library",
            index_type="flat",
            distance_metric="cosine",
        )
        print(f"✓ Created library: {library.id}")
        print(f"  Name: {library.name}")
        print(f"  Index Type: {library.index_config.index_type}")

        # 2. List all libraries
        print("\n2. Listing all libraries...")
        libraries = client.list_libraries(skip=0, limit=10)
        print(f"✓ Found {libraries.total} libraries")
        for lib in libraries.items:
            print(f"  - {lib.name} ({lib.id})")

        # 3. Create a document
        print("\n3. Creating a document...")
        document = client.create_document(
            library_id=library.id, name="Example Document"
        )
        print(f"✓ Created document: {document.id}")
        print(f"  Name: {document.name}")

        # 4. Add chunks with embeddings and metadata using Cohere
        print("\n4. Adding chunks with embeddings and metadata using Cohere...")
        chunks = []

        # Define texts with associated metadata
        # Note: Custom metadata must be nested under the "custom" key
        texts_with_metadata = [
            {
                "text": "The quick brown fox jumps over the lazy dog",
                "metadata": {"custom": {"category": "general", "language": "english", "source": "proverb"}}
            },
            {
                "text": "Python is a high-level programming language",
                "metadata": {"custom": {"category": "programming", "language": "english", "source": "technical"}}
            },
            {
                "text": "Vector databases enable semantic search",
                "metadata": {"custom": {"category": "database", "language": "english", "source": "technical"}}
            },
            {
                "text": "Machine learning models can understand text semantically",
                "metadata": {"custom": {"category": "ai", "language": "english", "source": "technical"}}
            },
            {
                "text": "Embeddings capture the meaning of words in vector space",
                "metadata": {"custom": {"category": "ai", "language": "english", "source": "technical"}}
            },
        ]

        # Generate embeddings for all texts at once
        print("  Generating embeddings...")
        texts = [item["text"] for item in texts_with_metadata]
        embeddings = generate_embeddings(texts, cohere_client)
        print(f"  Generated {len(embeddings)} embeddings (dimension: {len(embeddings[0])})")

        # Create chunks with embeddings and metadata
        for item, embedding in zip(texts_with_metadata, embeddings):
            chunk = client.create_chunk(
                document_id=document.id,
                text=item["text"],
                embedding=embedding,
                metadata=item["metadata"],
            )
            chunks.append(chunk)
            print(f"✓ Created chunk: {str(chunk.id)[:8]}... - {item['text'][:40]}... (category: {item['metadata']['custom']['category']})")

        # 5. Search for similar vectors using semantic search (no filters)
        print("\n5. Searching for similar vectors (no filters)...")
        query_text = "What is semantic understanding in AI?"
        print(f"  Query: '{query_text}'")

        # Generate embedding for the query
        query_response = cohere_client.embed(
            texts=[query_text],
            model="embed-english-v3.0",
            input_type="search_query",  # Use search_query for queries
        )
        query_vector = query_response.embeddings[0]

        results = client.search_library(
            library_id=library.id, query=query_vector, top_k=3
        )

        print(f"✓ Search completed")
        print(f"  Found {results.total_results} results:")
        for i, result in enumerate(results.results, 1):
            print(f"  {i}. Score: {result.score:.4f}")
            print(f"     Category: {result.chunk.metadata.custom.get('category', 'N/A')}")
            print(f"     Text: {result.chunk.text}")

        # 5b. Search with metadata filters
        print("\n5b. Searching with metadata filter (category='ai' only)...")
        print(f"  Query: '{query_text}'")
        print(f"  Filter: category = 'ai'")

        results_filtered = client.search_library(
            library_id=library.id,
            query=query_vector,
            top_k=3,
            filters={"category": "ai"}  # Only return results with category='ai'
        )

        print(f"✓ Search completed")
        print(f"  Found {results_filtered.total_results} results (filtered):")
        for i, result in enumerate(results_filtered.results, 1):
            print(f"  {i}. Score: {result.score:.4f}")
            print(f"     Category: {result.chunk.metadata.custom.get('category', 'N/A')}")
            print(f"     Text: {result.chunk.text}")

        # 6. Get document chunks
        print("\n6. Retrieving all chunks from document...")
        document_chunks = client.get_document_chunks(document_id=document.id)
        print(f"✓ Retrieved {len(document_chunks)} chunks")

        # 7. Delete a chunk and show updated search results
        print("\n7. Deleting a chunk and re-running search...")
        # Delete one of the AI category chunks
        chunk_to_delete = chunks[-1]  # "Embeddings capture the meaning..."
        print(f"  Deleting chunk: {chunk_to_delete.text[:50]}...")
        client.delete_chunk(chunk_to_delete.id)
        print(f"✓ Deleted chunk: {str(chunk_to_delete.id)[:8]}...")

        # Search again with the same query
        print(f"\n  Re-running search with same query...")
        results_after_delete = client.search_library(
            library_id=library.id,
            query=query_vector,
            top_k=3,
            filters={"category": "ai"}
        )

        print(f"✓ Search completed")
        print(f"  Found {results_after_delete.total_results} results (was {results_filtered.total_results} before deletion):")
        for i, result in enumerate(results_after_delete.results, 1):
            print(f"  {i}. Score: {result.score:.4f}")
            print(f"     Category: {result.chunk.metadata.custom.get('category', 'N/A')}")
            print(f"     Text: {result.chunk.text}")

        # Remove the deleted chunk from our list
        chunks.remove(chunk_to_delete)

        # 8. Update a document
        print("\n8. Updating document...")
        updated_document = client.update_document(
            document_id=document.id, name="Updated Example Document"
        )
        print(f"✓ Updated document name to: {updated_document.name}")

        # 9. Update a library
        print("\n9. Updating library...")
        updated_library = client.update_library(
            library_id=library.id, name="Updated SDK Example Library"
        )
        print(f"✓ Updated library name to: {updated_library.name}")

        # 10. Clean up - delete remaining resources
        print("\n10. Cleaning up remaining resources...")

        # Delete chunks
        for chunk in chunks:
            client.delete_chunk(chunk.id)
        print(f"✓ Deleted {len(chunks)} chunks")

        # Delete document
        client.delete_document(document.id)
        print(f"✓ Deleted document: {document.id}")

        # Delete library
        client.delete_library(library.id)
        print(f"✓ Deleted library: {library.id}")

        print("\n✓ Example completed successfully!")

    except NotFoundError as e:
        print(f"✗ Resource not found: {e.message}")
    except AlreadyExistsError as e:
        print(f"✗ Resource already exists: {e.message}")
    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        # Close the client
        client.close()
        print("\nClient connection closed.")


def context_manager_example():
    """Example using context manager."""
    print("\nContext Manager Example:")
    print("-" * 50)

    # Get configuration from environment
    api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")

    # Using the client as a context manager
    with VectorDBClient(base_url=api_base_url) as client:
        # Create library
        library = client.create_library(name="Context Manager Example")
        print(f"Created library: {library.name}")

        # List libraries
        libraries = client.list_libraries()
        print(f"Total libraries: {libraries.total}")

        # Clean up
        client.delete_library(library.id)
        print("Library deleted")

    # Client is automatically closed when exiting the context


def error_handling_example():
    """Example demonstrating error handling."""
    print("\nError Handling Example:")
    print("-" * 50)

    # Get configuration from environment
    api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")

    client = VectorDBClient(base_url=api_base_url)

    try:
        # Try to get a non-existent library
        client.get_library("00000000-0000-0000-0000-000000000000")
    except NotFoundError as e:
        print(f"✓ Caught NotFoundError: {e.message}")

    try:
        # Create a library
        library = client.create_library(name="Duplicate Test")

        # Try to create another library with the same name (if uniqueness is enforced)
        # This might not raise an error depending on your API implementation
        library2 = client.create_library(name="Duplicate Test")

        # Clean up
        client.delete_library(library.id)
        client.delete_library(library2.id)
    except AlreadyExistsError as e:
        print(f"✓ Caught AlreadyExistsError: {e.message}")
    finally:
        client.close()


if __name__ == "__main__":
    # Run main example
    main()

    # context_manager_example()
    # error_handling_example()
