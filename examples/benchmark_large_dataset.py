"""
Benchmark different index types with the IMDB movie reviews dataset.

This demonstrates how indexes scale with real-world data, focusing on
performance and sentiment accuracy with 1,000 balanced movie reviews.

Requirements:
    uv add cohere requests python-dotenv datasets

Usage:
    1. Copy .env.example to .env and add your Cohere API key
    2. uv run python examples/benchmark_large_dataset.py
"""

import os
import random
import time
from pathlib import Path
from typing import Any

import cohere
import requests
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not COHERE_API_KEY:
    print("Error: COHERE_API_KEY not set in .env file")
    print("Please copy .env.example to .env and add your Cohere API key")
    exit(1)

co = cohere.Client(COHERE_API_KEY)

# Dataset configuration
DATASET_NAME = "stanfordnlp/imdb"
DATASET_LIMIT = 1000  # Limit to 1000 reviews for faster benchmarking

print("=" * 80)
print(f"IMDB Movie Reviews Benchmark")
print("=" * 80)
print()


def load_imdb_dataset() -> tuple[list[str], list[int]]:
    """
    Load the IMDB movie reviews dataset (limited to DATASET_LIMIT, balanced 50/50).

    Returns:
        (texts, labels) tuple where labels are 0 (negative) or 1 (positive)
    """
    print(f"Loading IMDB dataset (limit: {DATASET_LIMIT:,} reviews, balanced 50/50)...")

    # Load train split (has 25,000 reviews total, balanced)
    dataset = load_dataset(DATASET_NAME, split="train")

    print(f"✓ Loaded {len(dataset):,} total reviews")

    # Separate into positive and negative reviews
    negative_reviews = []
    positive_reviews = []

    for sample in dataset:
        if sample["label"] == 0:
            negative_reviews.append(sample["text"][:500])
        else:
            positive_reviews.append(sample["text"][:500])

    # Take half from each sentiment
    samples_per_class = DATASET_LIMIT // 2

    texts = negative_reviews[:samples_per_class] + positive_reviews[:samples_per_class]
    labels = [0] * samples_per_class + [1] * samples_per_class

    print(f"✓ Selected {len(texts):,} reviews")
    print(f"  Label distribution: {samples_per_class:,} negative, {samples_per_class:,} positive")
    print()

    return texts, labels


def create_test_queries() -> tuple[list[str], list[int]]:
    """
    Create test queries with known sentiments for validation.

    Returns:
        (query_texts, expected_labels) where labels are 0 (negative) or 1 (positive)
    """
    queries = [
        # Positive queries (should match positive reviews)
        "This movie was absolutely amazing and I loved every minute of it. The acting was superb and the story was captivating.",
        "Incredible masterpiece with brilliant performances. One of the best films I have ever seen.",
        "Fantastic entertainment from start to finish. Highly recommend this excellent movie.",
        "Outstanding cinematography and powerful storytelling. A true work of art.",

        # Negative queries (should match negative reviews)
        "Terrible waste of time and money. This movie was boring and poorly made.",
        "Awful film with bad acting and a terrible script. Complete disappointment.",
        "Horrible movie, don't waste your time watching this garbage.",
        "The worst film I've ever seen. Terrible in every way possible.",
    ]

    # Expected labels: first 4 are positive (1), last 4 are negative (0)
    expected_labels = [1, 1, 1, 1, 0, 0, 0, 0]

    return queries, expected_labels


def get_embeddings_batched(texts: list[str], batch_size: int = 96) -> list[list[float]]:
    """Get embeddings in batches (Cohere has a 96 text limit)."""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"  Embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} ({len(batch)} texts)...")

        response = co.embed(
            texts=batch,
            model="embed-english-v3.0",
            input_type="search_document",
        )
        all_embeddings.extend(response.embeddings)
        time.sleep(0.2)  # Rate limiting

    return all_embeddings


def create_library(name: str, index_config: dict[str, Any]) -> str:
    response = requests.post(
        f"{API_BASE_URL}/libraries",
        json={
            "name": name,
            "index_config": index_config,
            "metadata": {
                "description": f"Large dataset benchmark: {index_config['index_type']}",
                "embedding_model": "cohere-embed-english-v3.0",
                "embedding_dimension": 1024,
            }
        }
    )
    response.raise_for_status()
    return response.json()["id"]


def add_documents(library_id: str, texts: list[str], embeddings: list[list[float]], labels: list[str]) -> None:
    doc_response = requests.post(
        f"{API_BASE_URL}/documents",
        params={"library_id": library_id},
        json={"name": "Benchmark Dataset"}
    )
    doc_response.raise_for_status()
    document_id = doc_response.json()["id"]

    # Add in batches
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        for j, (text, embedding, label) in enumerate(zip(
            texts[i:i + batch_size],
            embeddings[i:i + batch_size],
            labels[i:i + batch_size]
        )):
            chunk_response = requests.post(
                f"{API_BASE_URL}/chunks",
                params={"document_id": document_id},
                json={
                    "text": text,
                    "embedding": embedding,
                    "metadata": {
                        "label": label,
                        "index": i + j,
                    }
                }
            )
            chunk_response.raise_for_status()

        print(f"  Added chunks {i+1}-{min(i+batch_size, len(texts))} / {len(texts)}")


def search(library_id: str, query_embedding: list[float], k: int = 10) -> tuple[list[dict], float]:
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


def calculate_sentiment_accuracy(results: list[dict], expected_label: int, k: int = 10) -> float:
    """
    Calculate sentiment accuracy - how many results match the expected sentiment.

    Args:
        results: Search results
        expected_label: Expected label (0=negative, 1=positive)
        k: Number of results to check

    Returns:
        Accuracy score (0.0 to 1.0)
    """
    if not results:
        return 0.0

    matching = sum(
        1 for r in results[:k]
        if r["chunk"]["metadata"]["label"] == expected_label
    )
    return matching / min(k, len(results))


def run_benchmark():
    print("=" * 80)
    print("Large Dataset Benchmark (IMDB Movie Reviews)")
    print("=" * 80)
    print()

    # Step 1: Load IMDB dataset
    texts, labels = load_imdb_dataset()

    # Step 2: Create test queries
    query_texts, expected_labels = create_test_queries()
    print(f"Created {len(query_texts)} test queries:")
    print(f"  - {sum(expected_labels)} positive queries")
    print(f"  - {len(expected_labels) - sum(expected_labels)} negative queries")
    print()

    # Step 2: Generate embeddings
    print("Step 1: Generating embeddings for documents...")
    text_embeddings = get_embeddings_batched(texts)
    print(f"✓ Generated {len(text_embeddings)} document embeddings")
    print()

    print("Step 2: Generating query embeddings...")
    query_embeddings = co.embed(
        texts=query_texts,
        model="embed-english-v3.0",
        input_type="search_query",
    ).embeddings
    print(f"✓ Generated {len(query_embeddings)} query embeddings")
    print()

    # Configurations to test
    configurations = [
        {"index_type": "flat", "distance_metric": "cosine"},

        # LSH optimized for larger datasets
        {"index_type": "lsh", "distance_metric": "cosine", "n_hash_tables": 10, "n_hash_bits": 7},
        {"index_type": "lsh", "distance_metric": "cosine", "n_hash_tables": 15, "n_hash_bits": 7},

        # HNSW (should excel here)
        {"index_type": "hnsw", "distance_metric": "cosine", "M": 16, "ef_construction": 200, "ef_search": 100},
        {"index_type": "hnsw", "distance_metric": "cosine", "M": 32, "ef_construction": 400, "ef_search": 150},
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
        add_documents(library_id, texts, text_embeddings, labels)
        print(f"✓ Added {len(texts)} texts")

        # Run queries
        total_sentiment_accuracy = 0.0
        total_query_time = 0.0
        query_times = []

        for i, query_embedding in enumerate(query_embeddings):
            search_results, query_time = search(library_id, query_embedding, k=10)
            query_times.append(query_time)
            total_query_time += query_time

            # Calculate sentiment accuracy
            expected_label = expected_labels[i]
            sentiment_acc = calculate_sentiment_accuracy(search_results, expected_label, k=10)
            total_sentiment_accuracy += sentiment_acc

            sentiment_name = "positive" if expected_label == 1 else "negative"
            print(f"  Query {i+1} ({sentiment_name}): Sentiment={sentiment_acc:.0%}, Time={query_time*1000:.2f}ms")

        avg_sentiment_accuracy = total_sentiment_accuracy / len(query_embeddings)
        avg_query_time = total_query_time / len(query_embeddings)

        results.append({
            "config": config_name,
            "avg_sentiment_accuracy": avg_sentiment_accuracy,
            "avg_query_time_ms": avg_query_time * 1000,
            "min_query_time_ms": min(query_times) * 1000,
            "max_query_time_ms": max(query_times) * 1000,
        })

        print(f"✓ Average Sentiment Accuracy: {avg_sentiment_accuracy:.2%}")
        print(f"✓ Average Query Time: {avg_query_time*1000:.2f}ms")
        print()

    # Print summary
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()

    results_by_sentiment = sorted(results, key=lambda x: (-x["avg_sentiment_accuracy"], x["avg_query_time_ms"]))
    results_by_speed = sorted(results, key=lambda x: (x["avg_query_time_ms"], -x["avg_sentiment_accuracy"]))

    print("Ranked by Sentiment Accuracy:")
    print("-" * 85)
    print(f"{'Rank':<6} {'Configuration':<45} {'Sentiment':<15} {'Avg Time':<12}")
    print("-" * 85)
    for i, result in enumerate(results_by_sentiment, 1):
        print(f"{i:<6} {result['config']:<45} {result['avg_sentiment_accuracy']:.1%}{'':10} {result['avg_query_time_ms']:.2f}ms")
    print()

    print("Ranked by Speed:")
    print("-" * 85)
    print(f"{'Rank':<6} {'Configuration':<45} {'Avg Time':<15} {'Sentiment':<12}")
    print("-" * 85)
    for i, result in enumerate(results_by_speed, 1):
        print(f"{i:<6} {result['config']:<45} {result['avg_query_time_ms']:.2f}ms{'':7} {result['avg_sentiment_accuracy']:.1%}")
    print()

    print("=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()
    print(f"Dataset: IMDB Movie Reviews")
    print(f"Size: {len(texts):,} vectors (reviews)")
    print(f"Distribution: {len(texts)//2:,} negative + {len(texts)//2:,} positive (balanced)")
    print(f"Queries: {len(query_texts)} sentiment-labeled queries")
    print()
    print("Metrics:")
    print("  • Sentiment Accuracy: How many results match the query sentiment")
    print("  • Query Time: Average time to perform a search")
    print()
    print("Performance at this scale:")
    print("  • Flat index: Exact search, O(n) time complexity")
    print("  • HNSW: Approximate search, sublinear time, excellent accuracy")
    print("  • LSH: Fast approximate search, accuracy depends on parameters")
    print()

    best_sentiment = results_by_sentiment[0]
    fastest = results_by_speed[0]
    print(f"🎯 Best Sentiment Accuracy: {best_sentiment['config']}")
    print(f"   Sentiment Accuracy: {best_sentiment['avg_sentiment_accuracy']:.1%}")
    print(f"   Avg Time: {best_sentiment['avg_query_time_ms']:.2f}ms")
    print()
    print(f"⚡ Fastest: {fastest['config']}")
    print(f"   Avg Time: {fastest['avg_query_time_ms']:.2f}ms")
    print(f"   Sentiment Accuracy: {fastest['avg_sentiment_accuracy']:.1%}")
    print()


if __name__ == "__main__":
    try:
        run_benchmark()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
        print("Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
