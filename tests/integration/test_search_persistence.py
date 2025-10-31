"""Integration tests for vector_index persistence via REST API."""

import pytest
import httpx
import subprocess
import time
import tempfile
import shutil
from pathlib import Path
import os


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_port():
    """Return a test port that's unlikely to conflict."""
    return 18889


def start_server(data_dir: Path, port: int) -> subprocess.Popen:
    """Start the server with custom data directory."""
    env = os.environ.copy()
    env["DATA_DIR"] = str(data_dir)

    process = subprocess.Popen(
        ["uv", "run", "uvicorn", "vector_db.main:app",
         "--host", "127.0.0.1", "--port", str(port)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for server to be ready
    max_retries = 30
    for _ in range(max_retries):
        try:
            response = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
            if response.status_code == 200:
                return process
        except (httpx.ConnectError, httpx.ReadTimeout):
            time.sleep(0.5)

    # If we couldn't connect, kill process and raise
    process.terminate()
    raise RuntimeError("Server failed to start within timeout")


def stop_server(process: subprocess.Popen):
    """Stop the server gracefully."""
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def test_search_results_consistent_across_restart(temp_data_dir, test_port):
    """
    Integration test that verifies search results are consistent across server restarts.

    Creates multiple chunks with different embeddings, performs searches,
    restarts the server, and verifies the same searches return the same results.
    """
    server = None

    try:
        # Start server
        server = start_server(temp_data_dir, test_port)
        base_url = f"http://127.0.0.1:{test_port}"

        with httpx.Client(base_url=base_url, timeout=10.0) as client:
            # Create library
            library_response = client.post(
                "/api/v1/libraries",
                json={
                    "name": "Consistency Test Library",
                    "index_config": {
                        "index_type": "flat",
                        "distance_metric": "cosine"
                    }
                }
            )
            assert library_response.status_code == 201
            library_id = library_response.json()["id"]

            # Create document
            document_response = client.post(
                f"/api/v1/documents?library_id={library_id}",
                json={"name": "Test Document"}
            )
            assert document_response.status_code == 201
            document_id = document_response.json()["id"]

            # Create 10 chunks with distinct embeddings
            chunk_ids = []
            for i in range(10):
                chunk_response = client.post(
                    f"/api/v1/chunks?document_id={document_id}",
                    json={
                        "text": f"Chunk {i}",
                        "embedding": [float(i) * 0.1] * 128,
                        "metadata": {"index": i}
                    }
                )
                assert chunk_response.status_code == 201
                chunk_ids.append(chunk_response.json()["id"])

            # Perform searches with different query embeddings
            search_results = []
            for i in [0, 5, 9]:
                search_response = client.post(
                    f"/api/v1/libraries/{library_id}/search",
                    json={
                        "embedding": [float(i) * 0.1] * 128,
                        "k": 3
                    }
                )
                assert search_response.status_code == 200
                results = search_response.json()["results"]
                assert len(results) == 3
                search_results.append(results[0]["chunk"]["id"])

            # Sanity check: different queries should return different top results
            assert len(set(search_results)) > 1, \
                "Different search queries returned same top result"

        # Stop server
        stop_server(server)
        time.sleep(1)

        # Restart server
        server = start_server(temp_data_dir, test_port)

        with httpx.Client(base_url=base_url, timeout=10.0) as client:
            # Verify chunks still exist
            chunks_response = client.get(f"/api/v1/documents/{document_id}/chunks")
            assert chunks_response.status_code == 200
            chunks = chunks_response.json()
            assert len(chunks) == 10

            # Perform same searches after restart
            search_results_after = []
            for i in [0, 5, 9]:
                search_response = client.post(
                    f"/api/v1/libraries/{library_id}/search",
                    json={
                        "embedding": [float(i) * 0.1] * 128,
                        "k": 3
                    }
                )
                assert search_response.status_code == 200
                results = search_response.json()["results"]
                assert len(results) == 3
                search_results_after.append(results[0]["chunk"]["id"])

            # Verify search results are consistent after restart
            assert search_results == search_results_after, \
                f"Search results changed after restart! Before: {search_results}, After: {search_results_after}"

    finally:
        if server:
            stop_server(server)


def test_exact_match_search_across_restart(temp_data_dir, test_port):
    """
    Integration test that verifies exact embedding matches work correctly.

    Creates chunks with very distinct embeddings and verifies that:
    1. Each chunk can be uniquely identified by its embedding
    2. Search returns the correct chunk for each unique embedding
    3. This behavior persists across server restarts
    """
    server = None

    try:
        # Start server
        server = start_server(temp_data_dir, test_port)
        base_url = f"http://127.0.0.1:{test_port}"

        with httpx.Client(base_url=base_url, timeout=10.0) as client:
            # Create library
            library_response = client.post(
                "/api/v1/libraries",
                json={
                    "name": "Exact Match Test Library",
                    "index_config": {
                        "index_type": "flat",
                        "distance_metric": "cosine"
                    }
                }
            )
            assert library_response.status_code == 201
            library_id = library_response.json()["id"]

            # Create document
            document_response = client.post(
                f"/api/v1/documents?library_id={library_id}",
                json={"name": "Test Document"}
            )
            assert document_response.status_code == 201
            document_id = document_response.json()["id"]

            # Create chunks with VERY distinct embeddings
            # Using one-hot-like patterns to ensure each is uniquely identifiable
            chunk_data = [
                ("chunk_0", [1.0, 0.0, 0.0] * 42 + [1.0, 0.0]),
                ("chunk_1", [0.0, 1.0, 0.0] * 42 + [0.0, 1.0]),
                ("chunk_2", [0.0, 0.0, 1.0] * 42 + [0.0, 0.0]),
            ]

            chunk_id_map = {}  # text -> chunk_id
            for text, embedding in chunk_data:
                chunk_response = client.post(
                    f"/api/v1/chunks?document_id={document_id}",
                    json={
                        "text": text,
                        "embedding": embedding,
                        "metadata": {"name": text}
                    }
                )
                assert chunk_response.status_code == 201
                chunk_id_map[text] = chunk_response.json()["id"]

            # Search for each exact embedding - should return the matching chunk
            for text, embedding in chunk_data:
                search_response = client.post(
                    f"/api/v1/libraries/{library_id}/search",
                    json={
                        "embedding": embedding,
                        "k": 1
                    }
                )
                assert search_response.status_code == 200
                results = search_response.json()["results"]
                assert len(results) == 1

                top_result = results[0]["chunk"]
                assert top_result["id"] == chunk_id_map[text], \
                    f"Expected chunk '{text}' (ID: {chunk_id_map[text]}) to be top result, " \
                    f"but got ID: {top_result['id']}"

        # Restart server
        stop_server(server)
        time.sleep(1)
        server = start_server(temp_data_dir, test_port)

        with httpx.Client(base_url=base_url, timeout=10.0) as client:
            # Verify same searches work after restart
            for text, embedding in chunk_data:
                search_response = client.post(
                    f"/api/v1/libraries/{library_id}/search",
                    json={
                        "embedding": embedding,
                        "k": 1
                    }
                )
                assert search_response.status_code == 200
                results = search_response.json()["results"]
                assert len(results) == 1

                top_result = results[0]["chunk"]
                assert top_result["id"] == chunk_id_map[text], \
                    f"After restart: Expected chunk '{text}' (ID: {chunk_id_map[text]}) " \
                    f"to be top result, but got ID: {top_result['id']}"

    finally:
        if server:
            stop_server(server)
