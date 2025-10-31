# Vector Database Examples

This directory contains example scripts demonstrating how to use the Vector Database API with Cohere embeddings.

## Prerequisites

1. **Install dependencies:**
   ```bash
   # Install required dependencies for examples
   uv add cohere requests python-dotenv
   ```

   Note: `cohere`, `requests`, and `python-dotenv` are only needed to run the example scripts.

2. **Configure environment variables:**
   ```bash
   # Copy the example .env file
   cp .env.example .env

   # Edit .env and add your Cohere API key
   # Get your key from: https://dashboard.cohere.com/api-keys
   ```

   Edit `.env` file:
   ```bash
   COHERE_API_KEY="your_api_key_here"
   ```

3. **Start the Vector Database API server:**
   ```bash
   uv run uvicorn vector_db.main:app --reload --port 8000
   ```

## Examples

### 1. Simple Usage (`simple_usage.py`)

A straightforward example showing basic operations:
- Creating a library with HNSW index
- Embedding text with Cohere
- Adding documents and chunks
- Performing similarity searches

**Run:**
```bash
uv run python examples/simple_usage.py
```

**What it does:**
- Creates a knowledge base with 3 documents (AI, Science, Nature)
- Embeds 9 text chunks using Cohere
- Performs 3 semantic searches
- Displays top results with scores

**Expected output:**
```
Creating library...
✓ Created library: <uuid>

Adding documents and chunks...
✓ Created document: AI Overview
  → Added 3 chunks
✓ Created document: Science Facts
  → Added 3 chunks
...

Query: 'What is machine learning?'
Found 3 results:
  1. Score: 0.9245 | Distance: 0.0755
     Text: Machine learning is a subset of AI that learns from data.
     Document: AI Overview
...
```

### 2. Benchmark Indexes (`benchmark_indexes.py`)

Comprehensive benchmark comparing different index types and configurations:
- Tests 9 different index configurations
- Measures recall (accuracy) and query speed
- Provides recommendations based on your use case

**Run:**
```bash
uv run python examples/benchmark_indexes.py
```

**What it tests:**

| Index Type | Distance Metric | Parameters |
|------------|----------------|------------|
| Flat | Cosine, Euclidean, Dot Product | - |
| LSH | Cosine, Euclidean | tables=3/5, bits=4/8 |
| HNSW | Cosine, Euclidean | M=8/16, ef=20/50 |

**Metrics measured:**
- **Recall@5**: Percentage of relevant results in top 5
- **Query Time**: Average time per search (in milliseconds)
- **Balance Score**: Combined recall and speed metric

**Expected output:**
```
Vector Database Index Benchmark
================================================================================

Step 1: Generating embeddings...
✓ Generated 40 document embeddings
✓ Generated 5 query embeddings

Testing: flat-cosine
--------------------------------------------------------------------------------
✓ Created library: <uuid>
✓ Added 40 sentences
  Query 1: 'AI and machine learning are changing technology...'
    Expected: technology
    Recall@5: 100.00%
    Query time: 2.34ms
...

BENCHMARK RESULTS SUMMARY
================================================================================

Ranked by Recall:
Rank   Configuration                                      Recall@5    Avg Time
1      flat-cosine                                        95.00%      2.45ms
2      hnsw-cosine (M=16, ef=50)                         92.00%      1.87ms
3      lsh-cosine (tables=5, bits=8)                     78.00%      1.23ms
...

RECOMMENDATIONS
================================================================================

🏆 Best Recall: flat-cosine
   - Recall@5: 95.00%
   - Avg Query Time: 2.45ms

⚡ Fastest: lsh-cosine (tables=3, bits=4)
   - Avg Query Time: 0.98ms
   - Recall@5: 72.00%

⚖️  Best Balanced (Recall + Speed): hnsw-cosine (M=16, ef=50)
   - Recall@5: 92.00%
   - Avg Query Time: 1.87ms
   - Balance Score: 94.32%
```

## Understanding the Results

### Index Types

**Flat Index (Brute Force)**
- ✅ **Best recall** - Always finds the true nearest neighbors
- ❌ **Slower** - Compares query to all vectors
- 💡 **Use when**: Dataset is small (<10k vectors) or perfect accuracy is critical

**LSH Index (Locality-Sensitive Hashing)**
- ✅ **Fast** - Sublinear search time
- ⚠️ **Approximate** - May miss some relevant results
- ✅ **Memory efficient** - Good for very high-dimensional data
- 💡 **Use when**: Speed is priority and some accuracy loss is acceptable

**HNSW Index (Hierarchical Navigable Small World)**
- ✅ **Great balance** - High recall with fast queries
- ✅ **Scalable** - Handles large datasets well
- ⚠️ **More complex** - More parameters to tune
- 💡 **Use when**: Need production-ready performance with good accuracy

### Distance Metrics

**Cosine Distance**
- Best for normalized embeddings (Cohere, OpenAI, etc.)
- Measures angle between vectors, not magnitude
- Range: 0 (identical) to 2 (opposite)
- 💡 **Recommended for most use cases**

**Euclidean Distance**
- Standard geometric distance
- Sensitive to vector magnitude
- Range: 0 (identical) to ∞
- 💡 **Use when**: Vector magnitudes are meaningful

**Dot Product**
- Fast computation
- Assumes vectors are normalized
- Higher values = more similar
- 💡 **Use when**: Vectors are normalized and speed is critical

## Tuning Parameters

### LSH Parameters
- `n_hash_tables`: **Most important for recall!**
  - More tables = higher probability of finding similar vectors
  - Each table uses different random hyperplanes
  - A match in ANY table makes a vector a candidate
  - **Recommended**: Start with 10-15 tables for good recall
  - Trade-off: More tables = more memory + slightly slower queries
- `n_hash_bits`: Controls granularity (2^n_hash_bits = number of buckets per table)
  - **Critical**: Must match your dataset size!
  - Too many bits = too many buckets = data spread too thin = 0% recall
  - **Small datasets (<100 vectors)**: Use 4 bits (16 buckets)
  - **Medium datasets (100-10k vectors)**: Use 6-8 bits (64-256 buckets)
  - **Large datasets (>10k vectors)**: Use 8-10 bits (256-1024 buckets)
  - Rule of thumb: aim for ~5-10 vectors per bucket on average

### HNSW Parameters
- `M`: Connections per node
  - Higher = better recall, more memory
  - Recommended: 16 for most cases, 32 for high recall
- `ef_construction`: Quality during index building
  - Higher = better quality, slower build
  - Recommended: 200
- `ef_search`: Quality during search
  - Higher = better recall, slower search
  - Recommended: 50-100

## Tips

1. **Start simple**: Use the simple_usage.py example first to understand the basics

2. **Benchmark your data**: Use benchmark_indexes.py with your actual sentences to find the best configuration

3. **Cohere best practices**:
   - Use `input_type="search_document"` for documents
   - Use `input_type="search_query"` for queries
   - Use cosine distance (Cohere embeddings are normalized)

4. **Error handling**: Both examples include proper error handling and helpful error messages

5. **API documentation**: Visit http://localhost:8000/api/v1/docs for interactive API documentation

## Troubleshooting

**"Could not connect to the API server"**
- Make sure the server is running: `uv run uvicorn vector_db.main:app --reload --port 8000`
- Check the server is on port 8000: `curl http://localhost:8000/health`

**"COHERE_API_KEY not set in .env file"**
- Make sure you copied `.env.example` to `.env`
- Add your Cohere API key to the `.env` file
- Verify the file exists: `cat .env | grep COHERE_API_KEY`

**Slow performance**
- Try HNSW index with higher `ef_search` parameter
- Reduce dataset size for testing
- Check server logs for errors

**Low recall**
- Use Flat index for perfect recall
- Increase LSH `n_hash_tables` or HNSW `ef_search`
- Verify you're using the right distance metric (cosine for Cohere)
