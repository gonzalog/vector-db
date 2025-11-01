# Vector Database

A high-performance REST API for indexing and querying documents in a Vector Database, implementing custom vector indexing algorithms from scratch.

## Features

- **Custom Vector Indexes**: Three different indexing algorithms implemented from scratch
  - **Flat Index**: Brute-force exact nearest neighbor search
  - **LSH (Locality-Sensitive Hashing)**: Approximate nearest neighbor search with configurable hash tables
  - **HNSW (Hierarchical Navigable Small World)**: Graph-based approximate nearest neighbor search

- **Flexible Distance Metrics**: Support for cosine, euclidean, and dot product similarity
- **Metadata Filtering**: Filter search results by custom metadata
- **Persistent Storage**: SQLite database with vector storage in NumPy arrays
- **Python SDK**: Type-safe client library for easy integration
- **REST API**: FastAPI-based API with automatic OpenAPI documentation
- **Docker Support**: Production-ready containerization with Docker Compose

## Quick Start

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip

### Running from Source

1. **Clone the repository**
   ```bash
   git clone https://github.com/gonzalog/vector-db
   cd vector-db
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys (e.g., COHERE_API_KEY for embeddings)
   ```

3. **Install dependencies using uv**
   ```bash
   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install project dependencies
   uv sync
   ```

4. **Run the API server**
   ```bash
   uv run uvicorn vector_db.main:app --reload --port 8000
   ```

5. **Access the API**
   - API: http://localhost:8000/api/v1
   - Interactive docs: http://localhost:8000/api/v1/docs
   - Health check: http://localhost:8000/health

### Running with Docker

#### Production Mode

```bash
# Build and start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

#### Development Mode (with hot-reload)

```bash
# Start development service with source code mounted
docker-compose --profile dev up -d vector-db-dev

# View logs
docker-compose logs -f vector-db-dev

# Stop the service
docker-compose --profile dev down
```

The API will be available at http://localhost:8000/api/v1

## Project Structure

```
vector-db/
 src/
    vector_db/
        api/              # FastAPI route handlers
        core/             # Core functionality (persistence, settings)
        indexes/          # Vector index implementations
        models/           # Pydantic data models
        repositories/     # Data access layer
        sdk/              # Python SDK client
        services/         # Business logic layer
        main.py           # Application entry point
 tests/                    # Unit and integration tests
 examples/                 # Example scripts and usage
 Dockerfile               # Production container image
 docker-compose.yml       # Docker orchestration
 pyproject.toml           # Project dependencies
```

## Architecture and Design Choices

### 1. **Layered Architecture**

The project follows a clean layered architecture to separate concerns:

```
API Layer (FastAPI routes)
    |
Service Layer (Business logic)
    |
Repository Layer (Data access)
    |
Storage Layer (SQLite + NumPy + Pickle)
```

**Rationale**: This separation makes the code:
- **Testable**: Each layer can be tested independently
- **Maintainable**: Changes to storage don't affect API contracts
- **Flexible**: Easy to swap implementations (e.g., in-memory vs persistent storage)

### 2. **Custom Vector Index Implementations**

Rather than using external libraries like FAISS or Annoy, all vector indexes are implemented from scratch:

#### Flat Index
- **Algorithm**: Brute-force search comparing query to all vectors
- **Complexity**: O(n) for search
- **Use case**: Small datasets (<10k vectors) where exact results are critical
- **Trade-off**: Slow but guaranteed exact nearest neighbors

#### LSH (Locality-Sensitive Hashing)
- **Algorithm**: Random projection-based hashing with configurable tables and bits
- **Complexity**: O(n/b) where b is number of hash buckets
- **Use case**: Large datasets (10k-1M vectors) with moderate accuracy requirements
- **Trade-off**: Fast approximate search, configurable accuracy vs speed

#### HNSW (Hierarchical Navigable Small World)
- **Algorithm**: Multi-layer graph with greedy search and backtracking
- **Complexity**: O(log n) for search
- **Use case**: Large datasets (>100k vectors) requiring high accuracy
- **Trade-off**: Best balance of speed and accuracy, higher memory usage

**Rationale**:
- **Educational**: Understanding the internals of vector search
- **Control**: Fine-grained tuning for specific use cases
- **No external dependencies**: Reduces supply chain risk
- **Customizable**: Easy to add domain-specific optimizations

### 3. **Dual Storage Strategy**

The system uses a hybrid approach combining SQLite, NumPy arrays, and Pickle:

- **SQLite**: Stores metadata (library, document, chunk info)
- **NumPy (.npy files)**: Stores dense vector embeddings
- **Pickle (.pkl files)**: Stores vector index structures

**Rationale**:
- SQLite is excellent for relational metadata and ACID compliance
- NumPy arrays provide efficient numerical operations
- Pickle enables serialization of complex Python objects (index graphs)
- Separation allows independent scaling of each component

### 4. **In-Memory Cache with Persistence**

The repository layer maintains in-memory caches backed by persistent storage:

```python
class PersistentLibraryRepository:
    _libraries: dict[UUID, Library]  # In-memory cache
    _indexes: dict[UUID, VectorIndex]  # In-memory indexes
    _vectors: dict[UUID, np.ndarray]  # In-memory vectors
```

**Rationale**:
- **Performance**: Fast reads from memory for hot data
- **Durability**: All changes persisted to disk immediately
- **Startup**: Quick recovery by loading from disk on initialization
- **Trade-off**: Higher memory usage but much faster queries

### 5. **Concurrency Control**

Uses a two-level locking strategy:

```python
_global_lock: RLock            # For library creation/deletion
_library_locks: dict[UUID, ReadWriteLock]  # Per-library operations
```

- **Read locks**: Multiple concurrent searches allowed
- **Write locks**: Exclusive access for add/update/delete operations

**Rationale**:
- **Correctness**: Prevents race conditions in multi-threaded FastAPI
- **Performance**: Allows parallel searches on different libraries
- **Granularity**: Lock only what's needed, not entire database

### 6. **Type-Safe Python SDK**

Provides a first-class Python client with Pydantic models:

```python
client = VectorDBClient(base_url="http://localhost:8000")
library = client.create_library(name="My Library", index_type="hnsw")
results = client.search_library(library_id=library.id, query=embedding, top_k=10)
```

**Rationale**:
- **Developer Experience**: Auto-completion and type checking in IDEs
- **Error Handling**: Custom exception hierarchy for better debugging
- **Consistency**: Same models as API ensures type safety
- **Discoverability**: Clear method signatures vs raw HTTP calls

### 7. **Metadata Structure**

Custom metadata is nested under a `custom` key:

```python
metadata = {
    "custom": {
        "category": "ai",
        "source": "technical"
    }
}
```

**Rationale**:
- **Extensibility**: Reserve top-level keys for system metadata
- **Backwards compatibility**: Can add system fields without breaking user data
- **Clear separation**: User data isolated from internal metadata

### 8. **Asynchronous Design**

FastAPI routes and repository methods use async/await:

```python
async def create_library(library_create: LibraryCreate) -> Library:
    library_repo = get_library_repository()
    library = Library(name=library_create.name, ...)
    return await library_repo.create(library)
```

**Rationale**:
- **Scalability**: Handle many concurrent requests efficiently
- **I/O optimization**: Don't block on database operations
- **Future-proof**: Ready for async databases (PostgreSQL with asyncpg)

### 9. **Service Layer Pattern**

Business logic lives in service classes, not API routes:

```python
# Service handles complex operations
async def update_chunk(chunk_id: UUID, chunk_update: ChunkUpdate) -> Chunk:
    # Update database
    # Update document timestamp
    # Update vector index
    # All coordinated in one place
```

**Rationale**:
- **Reusability**: Services can be called from API, CLI, or background jobs
- **Testability**: Test business logic without HTTP overhead
- **Transaction-like behavior**: Coordinate multiple repository operations

### 10. **Docker Multi-Stage Builds**

Dockerfile uses multi-stage builds:

```dockerfile
FROM python:3.10-slim as builder
# Install dependencies
FROM python:3.10-slim
# Copy only runtime artifacts
```

**Rationale**:
- **Smaller images**: Builder tools not included in final image
- **Faster deployments**: Less data to transfer
- **Security**: Reduced attack surface

## API Usage

### Create a Library

```bash
curl -X POST "http://localhost:8000/api/v1/libraries" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Library",
    "index_type": "hnsw",
    "distance_metric": "cosine"
  }'
```

### Add a Document with Chunks

```bash
# Create document
curl -X POST "http://localhost:8000/api/v1/documents?library_id=<LIBRARY_ID>" \
  -H "Content-Type: application/json" \
  -d '{"name": "My Document"}'

# Add chunk with embedding
curl -X POST "http://localhost:8000/api/v1/chunks?document_id=<DOCUMENT_ID>" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Sample text",
    "embedding": [0.1, 0.2, ...],
    "metadata": {"custom": {"category": "example"}}
  }'
```

### Search

```bash
curl -X POST "http://localhost:8000/api/v1/libraries/<LIBRARY_ID>/search" \
  -H "Content-Type: application/json" \
  -d '{
    "embedding": [0.1, 0.2, ...],
    "k": 10,
    "metadata_filter": {"category": "example"}
  }'
```

## Python SDK Usage

See `examples/sdk_usage.py` for a comprehensive example. Quick snippet:

```python
from vector_db.sdk import VectorDBClient
import cohere

# Initialize clients
client = VectorDBClient(base_url="http://localhost:8000")
cohere_client = cohere.Client(api_key="your-key")

# Create library
library = client.create_library(
    name="My Library",
    index_type="hnsw",
    distance_metric="cosine"
)

# Create document
document = client.create_document(
    library_id=library.id,
    name="My Document"
)

# Generate embedding and add chunk
texts = ["Machine learning is fascinating"]
embeddings = cohere_client.embed(texts=texts, model="embed-english-v3.0").embeddings

chunk = client.create_chunk(
    document_id=document.id,
    text=texts[0],
    embedding=embeddings[0],
    metadata={"custom": {"category": "ai"}}
)

# Search with filters
query_embedding = cohere_client.embed(
    texts=["What is AI?"],
    model="embed-english-v3.0",
    input_type="search_query"
).embeddings[0]

results = client.search_library(
    library_id=library.id,
    query=query_embedding,
    top_k=5,
    filters={"category": "ai"}
)

for result in results.results:
    print(f"Score: {result.score:.4f} - {result.chunk.text}")
```

## Running Examples

```bash
# Simple usage example
uv run python examples/simple_usage.py

# Full SDK example with Cohere embeddings (requires COHERE_API_KEY in .env)
uv run python examples/sdk_usage.py

# Benchmark different index types
uv run python examples/benchmark_indexes.py

# Large dataset performance test
uv run python examples/benchmark_large_dataset.py
```

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/vector_db --cov-report=html

# Run specific test file
uv run pytest tests/indexes/test_flat_index.py -v

# Run integration tests only
uv run pytest tests/integration/ -v
```

## Environment Variables

See `.env.example` for all available configuration options:

## Data Persistence

All data is stored in the `data/` directory:

```
data/
 vector_db.sqlite    # SQLite database (metadata)
 vectors/            # NumPy arrays (embeddings)
    <library-id>.npy
 indexes/            # Pickled index structures
     <library-id>.pkl
```

To backup your data, simply copy the `data/` directory.
