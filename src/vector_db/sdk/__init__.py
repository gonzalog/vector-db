"""Vector DB Python SDK.

A Python client library for interacting with the Vector DB API.

Example:
    ```python
    from vector_db.sdk import VectorDBClient

    # Initialize client
    client = VectorDBClient(base_url="http://localhost:8000")

    # Create a library
    library = client.create_library(name="My Library")
    print(f"Created library: {library.id}")

    # Create a document
    document = client.create_document(
        library_id=library.id,
        name="My Document"
    )

    # Add chunks with embeddings
    chunk = client.create_chunk(
        document_id=document.id,
        text="Sample text",
        embedding=[0.1, 0.2, 0.3, ...]
    )

    # Search
    results = client.search_library(
        library_id=library.id,
        query=[0.1, 0.2, 0.3, ...],
        top_k=10
    )

    for result in results.results:
        print(f"Score: {result.score}, Text: {result.chunk.text}")
    ```
"""

from vector_db.sdk.client import VectorDBClient
from vector_db.sdk.exceptions import (
    AlreadyExistsError,
    ConnectionError,
    NotFoundError,
    ServerError,
    ValidationError,
    VectorDBSDKError,
)

__all__ = [
    "VectorDBClient",
    "VectorDBSDKError",
    "NotFoundError",
    "AlreadyExistsError",
    "ValidationError",
    "ServerError",
    "ConnectionError",
]
