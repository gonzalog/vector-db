"""Vector indexing package."""

from vector_db.indexes.base import VectorIndex
from vector_db.indexes.distance import (
    DistanceMetric,
    cosine_distance,
    cosine_similarity,
    dot_product,
    euclidean_distance,
    get_distance_function,
)
from vector_db.indexes.flat_index import FlatIndex
from vector_db.indexes.hnsw_index import HNSWIndex
from vector_db.indexes.lsh_index import LSHIndex

__all__ = [
    "VectorIndex",
    "FlatIndex",
    "LSHIndex",
    "HNSWIndex",
    "DistanceMetric",
    "cosine_similarity",
    "cosine_distance",
    "euclidean_distance",
    "dot_product",
    "get_distance_function",
]
