"""
Simple example of using the Vector Database API with Cohere embeddings.

This script demonstrates:
1. Creating a library with a specific index configuration
2. Embedding text using Cohere
3. Adding documents and chunks
4. Performing similarity searches

Requirements:
    uv pip install cohere requests python-dotenv

Usage:
    1. Copy .env.example to .env and add your Cohere API key
    2. uv run python examples/simple_usage.py
"""

import os
from pathlib import Path

import cohere
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not COHERE_API_KEY:
    print("Error: COHERE_API_KEY not set in .env file")
    print("Please copy .env.example to .env and add your Cohere API key")
    print("Get your key from: https://dashboard.cohere.com/api-keys")
    exit(1)

# Initialize Cohere client
co = cohere.Client(COHERE_API_KEY)

# Sample documents
documents_data = [
    {
        "name": "AI Overview",
        "chunks": [
            "Artificial intelligence is the simulation of human intelligence by machines.",
            "Machine learning is a subset of AI that learns from data.",
            "Deep learning uses neural networks with multiple layers.",
        ]
    },
    {
        "name": "Science Facts",
        "chunks": [
            "The speed of light is approximately 299,792 kilometers per second.",
            "DNA contains the genetic instructions for all living organisms.",
            "The periodic table organizes chemical elements by atomic number.",
        ]
    },
    {
        "name": "Nature",
        "chunks": [
            "Rainforests produce about 20% of the world's oxygen.",
            "The Amazon rainforest spans nine countries in South America.",
            "Coral reefs support about 25% of all marine species.",
        ]
    }
]


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings from Cohere API."""
    response = co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document",
    )
    return response.embeddings


def get_query_embedding(query: str) -> list[float]:
    """Get embedding for a search query."""
    response = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query",
    )
    return response.embeddings[0]


def main():
    print("=" * 80)
    print("Vector Database - Simple Usage Example")
    print("=" * 80)
    print()

    # Step 1: Create a library with HNSW index and cosine distance
    print("Step 1: Creating library...")
    library_response = requests.post(
        f"{API_BASE_URL}/libraries",
        json={
            "name": "My Knowledge Base",
            "index_config": {
                "index_type": "hnsw",
                "distance_metric": "cosine",
                "M": 16,
                "ef_construction": 200,
                "ef_search": 50,
            },
            "metadata": {
                "description": "A sample knowledge base using Cohere embeddings",
                "embedding_model": "cohere-embed-english-v3.0",
                "embedding_dimension": 1024,
            }
        }
    )
    library_response.raise_for_status()
    library_id = library_response.json()["id"]
    print(f"✓ Created library: {library_id}")
    print()

    # Step 2: Add documents with embeddings
    print("Step 2: Adding documents and chunks...")
    for doc_data in documents_data:
        # Create document
        doc_response = requests.post(
            f"{API_BASE_URL}/documents",
            params={"library_id": library_id},
            json={
                "name": doc_data["name"],
                "metadata": {
                    "source": "example_script"
                }
            }
        )
        doc_response.raise_for_status()
        document_id = doc_response.json()["id"]
        print(f"✓ Created document: {doc_data['name']}")

        # Get embeddings for all chunks at once
        chunk_embeddings = get_embeddings(doc_data["chunks"])

        # Add chunks with embeddings
        for chunk_text, embedding in zip(doc_data["chunks"], chunk_embeddings):
            chunk_response = requests.post(
                f"{API_BASE_URL}/chunks",
                params={"document_id": document_id},
                json={
                    "text": chunk_text,
                    "embedding": embedding,
                    "metadata": {
                        "document_name": doc_data["name"]
                    }
                }
            )
            chunk_response.raise_for_status()

        print(f"  → Added {len(doc_data['chunks'])} chunks")

    print()

    # Step 3: Perform searches
    print("Step 3: Performing searches...")
    print("-" * 80)

    queries = [
        "What is machine learning?",
        "Tell me about DNA",
        "Information about rainforests",
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        print()

        # Get query embedding
        query_embedding = get_query_embedding(query)

        # Search
        search_response = requests.post(
            f"{API_BASE_URL}/libraries/{library_id}/search",
            json={
                "embedding": query_embedding,
                "k": 3,
            }
        )
        search_response.raise_for_status()
        results = search_response.json()["results"]

        # Display results
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            chunk = result["chunk"]
            score = result["score"]
            distance = result["distance"]

            print(f"\n  {i}. Score: {score:.4f} | Distance: {distance:.4f}")
            print(f"     Text: {chunk['text']}")
            print(f"     Document: {chunk['metadata'].get('custom', {}).get('document_name', 'Unknown')}")

    print()
    print("=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
        print("Make sure the server is running on http://localhost:8000")
        print("Start it with: uv run uvicorn vector_db.main:app --reload --port 8000")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
