"""End-to-end test for API persistence across server restarts."""

import pytest
import httpx
import subprocess
import time
import tempfile
import shutil
from pathlib import Path
import signal
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
    return 18888


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


def test_vectors_persist_across_restart(temp_data_dir, test_port):
    """
    End-to-end test that verifies vectors persist across server restarts.

    This test:
    1. Starts a server
    2. Creates a library, document, and chunks with embeddings via HTTP API
    3. Performs a vector search
    4. Stops the server
    5. Starts a new server instance
    6. Verifies all data is still accessible
    7. Performs the same search to confirm vectors are properly indexed
    """
    server = None

    try:
        # Step 1: Start server
        server = start_server(temp_data_dir, test_port)
        base_url = f"http://127.0.0.1:{test_port}"

        # Step 2: Create library via API
        with httpx.Client(base_url=base_url, timeout=10.0) as client:
            # Create library
            library_response = client.post(
                "/api/v1/libraries",
                json={
                    "name": "Persistence Test Library",
                    "metadata": {
                        "description": "Testing persistence across restarts"
                    },
                    "index_config": {
                        "index_type": "flat",
                        "distance_metric": "cosine"
                    }
                }
            )
            assert library_response.status_code == 201
            library_data = library_response.json()
            library_id = library_data["id"]

            # Create document
            document_response = client.post(
                f"/api/v1/documents?library_id={library_id}",
                json={
                    "name": "Test Document",
                    "metadata": {
                        "title": "Persistence Test"
                    }
                }
            )
            assert document_response.status_code == 201
            document_data = document_response.json()
            document_id = document_data["id"]

            # Create chunks with embeddings
            chunk_texts = [
                "The quick brown fox jumps over the lazy dog",
                "Machine learning is a subset of artificial intelligence",
                "Vector databases enable semantic search capabilities"
            ]

            chunk_ids = []
            for i, text in enumerate(chunk_texts):
                # Create simple embeddings (in real use, these would come from an embedding model)
                embedding = [float(i) * 0.1] * 128

                chunk_response = client.post(
                    f"/api/v1/chunks?document_id={document_id}",
                    json={
                        "text": text,
                        "embedding": embedding,
                        "metadata": {
                            "index": i
                        }
                    }
                )
                assert chunk_response.status_code == 201
                chunk_data = chunk_response.json()
                chunk_ids.append(chunk_data["id"])

            # Step 3: Perform vector search
            search_response = client.post(
                f"/api/v1/libraries/{library_id}/search",
                json={
                    "embedding": [0.1] * 128,  # Similar to chunk 1
                    "k": 3
                }
            )
            assert search_response.status_code == 200
            search_data = search_response.json()
            assert search_data["total_results"] == 3
            initial_results = search_data["results"]

            # Verify we got results
            assert len(initial_results) > 0
            initial_top_result_id = initial_results[0]["chunk"]["id"]

        # Step 4: Stop server
        stop_server(server)
        time.sleep(1)  # Give it time to fully shut down

        # Step 5: Restart server with same data directory
        server = start_server(temp_data_dir, test_port)

        # Step 6 & 7: Verify data persisted
        with httpx.Client(base_url=base_url, timeout=10.0) as client:
            # Verify library still exists
            library_get_response = client.get(f"/api/v1/libraries/{library_id}")
            assert library_get_response.status_code == 200
            library_get_data = library_get_response.json()
            assert library_get_data["id"] == library_id
            assert library_get_data["name"] == "Persistence Test Library"

            # Verify document still exists
            document_get_response = client.get(f"/api/v1/documents/{document_id}")
            assert document_get_response.status_code == 200
            document_get_data = document_get_response.json()
            assert document_get_data["id"] == document_id
            assert document_get_data["name"] == "Test Document"

            # Verify chunks still exist
            chunks_response = client.get(f"/api/v1/documents/{document_id}/chunks")
            assert chunks_response.status_code == 200
            chunks_data = chunks_response.json()
            assert len(chunks_data) == 3

            # Verify chunk texts match
            chunk_texts_retrieved = sorted([chunk["text"] for chunk in chunks_data])
            assert chunk_texts_retrieved == sorted(chunk_texts)

            # Step 7: Perform same search to verify index was reconstructed
            search_response_after = client.post(
                f"/api/v1/libraries/{library_id}/search",
                json={
                    "embedding": [0.1] * 128,
                    "k": 3
                }
            )
            assert search_response_after.status_code == 200
            search_data_after = search_response_after.json()
            assert search_data_after["total_results"] == 3

            # Verify search results are consistent
            results_after = search_data_after["results"]
            assert len(results_after) == 3

            # The top result should be the same (same embedding should match same chunk)
            top_result_after_id = results_after[0]["chunk"]["id"]
            assert top_result_after_id == initial_top_result_id

            # Verify all chunk IDs are present in results
            result_chunk_ids = {result["chunk"]["id"] for result in results_after}
            assert result_chunk_ids == set(chunk_ids)

    finally:
        # Cleanup
        if server:
            stop_server(server)


def test_multiple_operations_persist_across_restart(temp_data_dir, test_port):
    """
    Test that multiple operations (create, update, delete) persist correctly.
    """
    server = None

    try:
        # Start server
        server = start_server(temp_data_dir, test_port)
        base_url = f"http://127.0.0.1:{test_port}"

        with httpx.Client(base_url=base_url, timeout=10.0) as client:
            # Create 3 libraries
            library_ids = []
            for i in range(3):
                response = client.post(
                    "/api/v1/libraries",
                    json={"name": f"Library {i}"}
                )
                assert response.status_code == 201
                library_ids.append(response.json()["id"])

            # Delete the middle one
            deleted_id = library_ids[1]
            delete_response = client.delete(f"/api/v1/libraries/{deleted_id}")
            assert delete_response.status_code == 204

            # Update the first one
            updated_id = library_ids[0]
            update_response = client.put(
                f"/api/v1/libraries/{updated_id}",
                json={"name": "Updated Library Name"}
            )
            assert update_response.status_code == 200

        # Restart server
        stop_server(server)
        time.sleep(1)
        server = start_server(temp_data_dir, test_port)

        with httpx.Client(base_url=base_url, timeout=10.0) as client:
            # Verify deleted library doesn't exist
            response = client.get(f"/api/v1/libraries/{deleted_id}")
            assert response.status_code == 404

            # Verify updated library has new name
            response = client.get(f"/api/v1/libraries/{updated_id}")
            assert response.status_code == 200
            assert response.json()["name"] == "Updated Library Name"

            # Verify third library still exists
            response = client.get(f"/api/v1/libraries/{library_ids[2]}")
            assert response.status_code == 200
            assert response.json()["name"] == "Library 2"

    finally:
        if server:
            stop_server(server)
