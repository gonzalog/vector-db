"""
Benchmark different index types and distance metrics using Cohere embeddings.

This script:
1. Uses Cohere API to generate embeddings for a set of sentences
2. Creates libraries with different index configurations
3. Measures recall and performance for each configuration
4. Provides recommendations based on the results

Requirements:
    uv pip install cohere requests python-dotenv

Usage:
    1. Copy .env.example to .env and add your Cohere API key
    2. uv run python examples/benchmark_indexes.py
"""

import os
import time
from pathlib import Path
from typing import Any

import cohere
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not COHERE_API_KEY:
    print("Error: COHERE_API_KEY not set in .env file")
    print("Please copy .env.example to .env and add your Cohere API key")
    print("Get your key from: https://dashboard.cohere.com/api-keys")
    exit(1)

# Initialize Cohere client
co = cohere.Client(COHERE_API_KEY)

# Sample sentences for testing
SENTENCES = [
    # Technology
    "Artificial intelligence is transforming the way we work and live.",
    "Machine learning algorithms can predict patterns in large datasets.",
    "Deep learning models require significant computational resources.",
    "Natural language processing enables computers to understand human language.",
    "Computer vision allows machines to interpret visual information.",

    # Science
    "The theory of relativity revolutionized our understanding of space and time.",
    "Quantum mechanics describes the behavior of matter at atomic scales.",
    "DNA carries the genetic instructions for all living organisms.",
    "Evolution by natural selection explains the diversity of life on Earth.",
    "The periodic table organizes elements by their chemical properties.",

    # Nature
    "Rainforests are home to incredible biodiversity and produce oxygen.",
    "Climate change is affecting ecosystems around the world.",
    "Coral reefs are complex underwater ecosystems threatened by warming oceans.",
    "Mountains are formed by tectonic plate movements over millions of years.",
    "Rivers carve landscapes and provide fresh water to communities.",

    # Arts
    "Music has the power to evoke emotions and create connections.",
    "Painting allows artists to express their vision through color and form.",
    "Literature captures human experiences and preserves cultural heritage.",
    "Dance combines movement and rhythm to tell stories without words.",
    "Sculpture transforms raw materials into three-dimensional art.",

    # Food
    "Italian cuisine is known for its pasta, pizza, and rich flavors.",
    "Japanese food emphasizes fresh ingredients and beautiful presentation.",
    "Mediterranean diets include olive oil, vegetables, and seafood.",
    "Baking bread requires precision in measuring ingredients and timing.",
    "Fermentation preserves food and creates unique flavors like in cheese.",

    # Sports
    "Soccer is the most popular sport worldwide with billions of fans.",
    "Basketball combines athleticism, strategy, and teamwork.",
    "Swimming is an excellent full-body workout and competitive sport.",
    "Tennis requires agility, precision, and mental focus.",
    "Marathon running tests endurance and determination.",

    # History
    "The Roman Empire shaped Western civilization and law.",
    "The Industrial Revolution transformed societies with mechanization.",
    "Ancient Egypt built magnificent pyramids and developed hieroglyphics.",
    "The Renaissance was a period of cultural rebirth in Europe.",
    "World War II changed the global political landscape.",

    # Travel
    "Paris is famous for the Eiffel Tower and its romantic atmosphere.",
    "Tokyo blends traditional culture with modern technology.",
    "The Amazon rainforest offers unparalleled biodiversity experiences.",
    "Antarctica is the coldest and most remote continent on Earth.",
    "The Great Wall of China stretches thousands of miles.",
]

# Query sentences to test recall
QUERY_SENTENCES = [
    "AI and machine learning are changing technology",  # Should match tech
    "DNA and genetics explain heredity",  # Should match science
    "Forests produce oxygen and support wildlife",  # Should match nature
    "Paintings and visual art express creativity",  # Should match arts
    "Pasta and Mediterranean cooking traditions",  # Should match food
]

# Expected categories for queries (for recall calculation)
EXPECTED_CATEGORIES = {
    0: "technology",
    1: "science",
    2: "nature",
    3: "arts",
    4: "food",
}

# Sentence categories (for recall validation)
SENTENCE_CATEGORIES = (
    ["technology"] * 5 +
    ["science"] * 5 +
    ["nature"] * 5 +
    ["arts"] * 5 +
    ["food"] * 5 +
    ["sports"] * 5 +
    ["history"] * 5 +
    ["travel"] * 5
)


def get_embeddings(texts: list[str], model: str = "embed-english-v3.0") -> list[list[float]]:
    """Get embeddings from Cohere API."""
    print(f"Getting embeddings for {len(texts)} texts using {model}...")
    response = co.embed(
        texts=texts,
        model=model,
        input_type="search_document",
    )
    return response.embeddings


def create_library(name: str, index_config: dict[str, Any]) -> str:
    """Create a library with specified index configuration."""
    response = requests.post(
        f"{API_BASE_URL}/libraries",
        json={
            "name": name,
            "index_config": index_config,
            "metadata": {
                "description": f"Benchmark library for {index_config['index_type']} with {index_config['distance_metric']}",
                "embedding_model": "cohere-embed-english-v3.0",
                "embedding_dimension": 1024,
            }
        }
    )
    response.raise_for_status()
    return response.json()["id"]


def add_documents(library_id: str, sentences: list[str], embeddings: list[list[float]]) -> str:
    """Add documents and chunks to library."""
    # Create a single document
    doc_response = requests.post(
        f"{API_BASE_URL}/documents",
        params={"library_id": library_id},
        json={"name": "Benchmark Sentences"}
    )
    doc_response.raise_for_status()
    document_id = doc_response.json()["id"]

    # Add all sentences as chunks
    for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
        chunk_response = requests.post(
            f"{API_BASE_URL}/chunks",
            params={"document_id": document_id},
            json={
                "text": sentence,
                "embedding": embedding,
                "metadata": {
                    "category": SENTENCE_CATEGORIES[i],
                    "index": i,
                }
            }
        )
        chunk_response.raise_for_status()

    return document_id


def search(library_id: str, query_embedding: list[float], k: int = 5) -> tuple[list[dict], float]:
    """Perform search and return results with query time."""
    start_time = time.time()
    response = requests.post(
        f"{API_BASE_URL}/libraries/{library_id}/search",
        json={
            "embedding": query_embedding,
            "k": k,
        }
    )
    query_time = time.time() - start_time
    response.raise_for_status()
    return response.json()["results"], query_time


def calculate_recall(results: list[dict], expected_category: str, k: int = 5) -> float:
    """Calculate recall - how many results match the expected category."""
    matching = sum(
        1 for result in results[:k]
        if result["chunk"]["metadata"].get("custom", {}).get("category") == expected_category
    )
    return matching / k


def run_benchmark():
    """Run comprehensive benchmark of all index configurations."""
    print("=" * 80)
    print("Vector Database Index Benchmark")
    print("=" * 80)
    print()

    # Get embeddings for all sentences
    print("Step 1: Generating embeddings...")
    sentence_embeddings = get_embeddings(SENTENCES)
    query_embeddings = get_embeddings(QUERY_SENTENCES, model="embed-english-v3.0")
    # Change input type for queries
    query_embeddings = co.embed(
        texts=QUERY_SENTENCES,
        model="embed-english-v3.0",
        input_type="search_query",
    ).embeddings
    print(f"✓ Generated {len(sentence_embeddings)} document embeddings")
    print(f"✓ Generated {len(query_embeddings)} query embeddings")
    print()

    # Index configurations to test
    configurations = [
        # Flat index with different distance metrics
        {"index_type": "flat", "distance_metric": "cosine"},
        {"index_type": "flat", "distance_metric": "euclidean"},
        {"index_type": "flat", "distance_metric": "dot_product"},

        # LSH index with different configurations
        {"index_type": "lsh", "distance_metric": "cosine", "n_hash_tables": 3, "n_hash_bits": 4},
        {"index_type": "lsh", "distance_metric": "cosine", "n_hash_tables": 5, "n_hash_bits": 8},
        {"index_type": "lsh", "distance_metric": "euclidean", "n_hash_tables": 5, "n_hash_bits": 8},

        # HNSW index with different configurations
        {"index_type": "hnsw", "distance_metric": "cosine", "M": 8, "ef_construction": 50, "ef_search": 20},
        {"index_type": "hnsw", "distance_metric": "cosine", "M": 16, "ef_construction": 200, "ef_search": 50},
        {"index_type": "hnsw", "distance_metric": "euclidean", "M": 16, "ef_construction": 200, "ef_search": 50},
    ]

    results = []

    for config in configurations:
        config_name = f"{config['index_type']}-{config['distance_metric']}"
        if config['index_type'] == 'lsh':
            config_name += f" (tables={config['n_hash_tables']}, bits={config['n_hash_bits']})"
        elif config['index_type'] == 'hnsw':
            config_name += f" (M={config['M']}, ef={config['ef_search']})"

        print(f"Testing: {config_name}")
        print("-" * 80)

        # Create library
        library_id = create_library(f"Benchmark-{config_name}", config)
        print(f"✓ Created library: {library_id}")

        # Add documents
        add_documents(library_id, SENTENCES, sentence_embeddings)
        print(f"✓ Added {len(SENTENCES)} sentences")

        # Run queries and measure performance
        total_recall = 0.0
        total_query_time = 0.0
        query_times = []

        for i, (query_text, query_embedding) in enumerate(zip(QUERY_SENTENCES, query_embeddings)):
            search_results, query_time = search(library_id, query_embedding, k=5)
            query_times.append(query_time)
            total_query_time += query_time

            expected_category = EXPECTED_CATEGORIES[i]
            recall = calculate_recall(search_results, expected_category, k=5)
            total_recall += recall

            print(f"  Query {i+1}: '{query_text[:50]}...'")
            print(f"    Expected: {expected_category}")
            print(f"    Recall@5: {recall:.2%}")
            print(f"    Query time: {query_time*1000:.2f}ms")

        avg_recall = total_recall / len(QUERY_SENTENCES)
        avg_query_time = total_query_time / len(QUERY_SENTENCES)

        results.append({
            "config": config_name,
            "config_dict": config,
            "avg_recall": avg_recall,
            "avg_query_time_ms": avg_query_time * 1000,
            "min_query_time_ms": min(query_times) * 1000,
            "max_query_time_ms": max(query_times) * 1000,
        })

        print(f"✓ Average Recall@5: {avg_recall:.2%}")
        print(f"✓ Average Query Time: {avg_query_time*1000:.2f}ms")
        print()

    # Print summary
    print("=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print()

    # Sort by recall (descending), then by query time (ascending)
    results_by_recall = sorted(results, key=lambda x: (-x["avg_recall"], x["avg_query_time_ms"]))
    results_by_speed = sorted(results, key=lambda x: (x["avg_query_time_ms"], -x["avg_recall"]))

    print("Ranked by Recall:")
    print("-" * 80)
    print(f"{'Rank':<6} {'Configuration':<50} {'Recall@5':<12} {'Avg Time':<12}")
    print("-" * 80)
    for i, result in enumerate(results_by_recall, 1):
        print(f"{i:<6} {result['config']:<50} {result['avg_recall']:.2%}{'':6} {result['avg_query_time_ms']:.2f}ms")
    print()

    print("Ranked by Speed:")
    print("-" * 80)
    print(f"{'Rank':<6} {'Configuration':<50} {'Avg Time':<12} {'Recall@5':<12}")
    print("-" * 80)
    for i, result in enumerate(results_by_speed, 1):
        print(f"{i:<6} {result['config']:<50} {result['avg_query_time_ms']:.2f}ms{'':4} {result['avg_recall']:.2%}")
    print()

    # Recommendations
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    best_recall = results_by_recall[0]
    fastest = results_by_speed[0]

    print(f"🏆 Best Recall: {best_recall['config']}")
    print(f"   - Recall@5: {best_recall['avg_recall']:.2%}")
    print(f"   - Avg Query Time: {best_recall['avg_query_time_ms']:.2f}ms")
    print()

    print(f"⚡ Fastest: {fastest['config']}")
    print(f"   - Avg Query Time: {fastest['avg_query_time_ms']:.2f}ms")
    print(f"   - Recall@5: {fastest['avg_recall']:.2%}")
    print()

    # Find best balanced option
    # Calculate score: normalize recall (0-1) and speed (inverse, 0-1), then average
    max_time = max(r["avg_query_time_ms"] for r in results)
    for r in results:
        # Higher is better for both (recall is already 0-1, speed is inverted)
        speed_score = 1 - (r["avg_query_time_ms"] / max_time)
        r["balance_score"] = (r["avg_recall"] + speed_score) / 2

    best_balanced = max(results, key=lambda x: x["balance_score"])
    print(f"⚖️  Best Balanced (Recall + Speed): {best_balanced['config']}")
    print(f"   - Recall@5: {best_balanced['avg_recall']:.2%}")
    print(f"   - Avg Query Time: {best_balanced['avg_query_time_ms']:.2f}ms")
    print(f"   - Balance Score: {best_balanced['balance_score']:.2%}")
    print()

    print("General Guidelines:")
    print("  • Flat index: Best recall, slower for large datasets")
    print("  • LSH index: Good for high-dimensional data, approximate results")
    print("  • HNSW index: Best balance of speed and recall for most use cases")
    print("  • Cosine distance: Best for normalized embeddings (recommended for Cohere)")
    print("  • Euclidean distance: Good for unnormalized vectors")
    print("  • Dot product: Fast but assumes normalized vectors")
    print()

    print("Dataset size: {} sentences".format(len(SENTENCES)))
    print("Embedding model: Cohere embed-english-v3.0")
    print("Embedding dimension: 1024")
    print()


if __name__ == "__main__":
    try:
        run_benchmark()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
        print("Make sure the server is running on http://localhost:8000")
        print("Start it with: uv run uvicorn vector_db.main:app --reload --port 8000")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
