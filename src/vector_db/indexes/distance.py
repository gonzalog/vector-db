"""Distance and similarity metrics for vector operations."""

from enum import Enum
from typing import Union

import numpy as np
import numpy.typing as npt


class DistanceMetric(str, Enum):
    """Supported distance metrics."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


def cosine_similarity(
    vec1: Union[list[float], npt.NDArray[np.float32]],
    vec2: Union[list[float], npt.NDArray[np.float32]],
) -> float:
    """
    Calculate cosine similarity between two vectors using NumPy.

    Args:
        vec1: First vector (list or numpy array)
        vec2: Second vector (list or numpy array)

    Returns:
        Cosine similarity (1 = identical, 0 = orthogonal, -1 = opposite)

    Raises:
        ValueError: If vectors have different dimensions
    """
    v1 = np.asarray(vec1, dtype=np.float32)
    v2 = np.asarray(vec2, dtype=np.float32)

    if v1.shape != v2.shape:
        raise ValueError(f"Vectors must have same dimension: {v1.shape} != {v2.shape}")

    # Compute dot product and norms
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    # Handle zero vectors by returning 0.0 similarity
    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot / (norm1 * norm2))


def cosine_distance(
    vec1: Union[list[float], npt.NDArray[np.float32]],
    vec2: Union[list[float], npt.NDArray[np.float32]],
) -> float:
    """
    Calculate cosine distance between two vectors.

    Args:
        vec1: First vector (list or numpy array)
        vec2: Second vector (list or numpy array)

    Returns:
        Cosine distance (0 = identical, 2 = opposite)
    """
    return 1.0 - cosine_similarity(vec1, vec2)


def euclidean_distance(
    vec1: Union[list[float], npt.NDArray[np.float32]],
    vec2: Union[list[float], npt.NDArray[np.float32]],
) -> float:
    """
    Calculate Euclidean (L2) distance between two vectors using NumPy.

    Args:
        vec1: First vector (list or numpy array)
        vec2: Second vector (list or numpy array)

    Returns:
        Euclidean distance (0 = identical)

    Raises:
        ValueError: If vectors have different dimensions
    """
    v1 = np.asarray(vec1, dtype=np.float32)
    v2 = np.asarray(vec2, dtype=np.float32)

    if v1.shape != v2.shape:
        raise ValueError(f"Vectors must have same dimension: {v1.shape} != {v2.shape}")

    return float(np.linalg.norm(v1 - v2))


def dot_product(
    vec1: Union[list[float], npt.NDArray[np.float32]],
    vec2: Union[list[float], npt.NDArray[np.float32]],
) -> float:
    """
    Calculate dot product between two vectors using NumPy.

    Args:
        vec1: First vector (list or numpy array)
        vec2: Second vector (list or numpy array)

    Returns:
        Dot product (higher = more similar for normalized vectors)

    Raises:
        ValueError: If vectors have different dimensions
    """
    v1 = np.asarray(vec1, dtype=np.float32)
    v2 = np.asarray(vec2, dtype=np.float32)

    if v1.shape != v2.shape:
        raise ValueError(f"Vectors must have same dimension: {v1.shape} != {v2.shape}")

    return float(np.dot(v1, v2))


def compute_batch_distances(
    query: Union[list[float], npt.NDArray[np.float32]],
    embeddings: Union[list[list[float]], npt.NDArray[np.float32]],
    metric: DistanceMetric,
) -> npt.NDArray[np.float32]:
    """
    Compute distances from a query vector to multiple embeddings using vectorized operations.

    Args:
        query: Query embedding vector
        embeddings: List or array of embeddings to compare against
        metric: Distance metric to use

    Returns:
        NumPy array of distances (lower is better for all metrics)
    """
    query_vec = np.asarray(query, dtype=np.float32)
    embeddings_array = np.asarray(embeddings, dtype=np.float32)

    if metric == DistanceMetric.COSINE:
        # Cosine distance: 1 - cosine_similarity
        dots = np.dot(embeddings_array, query_vec)
        norms_embeddings = np.linalg.norm(embeddings_array, axis=1)
        norm_query = np.linalg.norm(query_vec)
        similarities = dots / (norms_embeddings * norm_query + 1e-8)
        return 1.0 - similarities

    elif metric == DistanceMetric.EUCLIDEAN:
        # Euclidean distance
        return np.linalg.norm(embeddings_array - query_vec, axis=1)

    else:  # DOT_PRODUCT
        # Negative dot product (lower is better)
        return -np.dot(embeddings_array, query_vec)


def compute_distance_matrix(
    data: npt.NDArray[np.float32],
    centroids: npt.NDArray[np.float32],
    metric: DistanceMetric,
) -> npt.NDArray[np.float32]:
    """
    Compute distance matrix between two sets of vectors.

    Args:
        data: Array of data points, shape (n_samples, n_features)
        centroids: Array of centroids, shape (n_centroids, n_features)
        metric: Distance metric to use

    Returns:
        Distance matrix, shape (n_samples, n_centroids)
    """
    n_samples = data.shape[0]
    n_centroids = centroids.shape[0]
    distances = np.zeros((n_samples, n_centroids), dtype=np.float32)

    for i in range(n_centroids):
        if metric == DistanceMetric.COSINE:
            # Cosine distance: 1 - cosine_similarity
            dots = np.dot(data, centroids[i])
            norms_data = np.linalg.norm(data, axis=1)
            norm_centroid = np.linalg.norm(centroids[i])
            similarities = dots / (norms_data * norm_centroid + 1e-8)
            distances[:, i] = 1.0 - similarities

        elif metric == DistanceMetric.EUCLIDEAN:
            # Euclidean distance
            distances[:, i] = np.linalg.norm(data - centroids[i], axis=1)

        else:  # DOT_PRODUCT
            # Negative dot product (lower is better)
            distances[:, i] = -np.dot(data, centroids[i])

    return distances


def get_distance_function(metric: DistanceMetric):
    """
    Get the distance function for a given metric.

    Args:
        metric: The distance metric to use

    Returns:
        Distance function that takes two vectors and returns a float

    Note:
        For metrics where higher is better (cosine_similarity, dot_product),
        returns the negative to maintain consistent ordering (lower = better).
    """
    if metric == DistanceMetric.COSINE:
        return cosine_distance
    elif metric == DistanceMetric.EUCLIDEAN:
        return euclidean_distance
    elif metric == DistanceMetric.DOT_PRODUCT:
        # Return negative dot product so lower is better
        return lambda v1, v2: -dot_product(v1, v2)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")
