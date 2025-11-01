"""HNSW (Hierarchical Navigable Small World) vector index implementation."""

import heapq
from typing import Any
from uuid import UUID

import numpy as np

from vector_db.core.locks import ReadWriteLock
from vector_db.indexes.base import VectorIndex
from vector_db.indexes.distance import DistanceMetric, get_distance_function
from vector_db.models import Chunk


class GraphLayer:
    """
    Represents a single layer in the HNSW graph structure.

    Each layer maintains bidirectional connections between nodes.
    """

    def __init__(self):
        """Initialize an empty graph layer."""
        self._connections: dict[UUID, list[UUID]] = {}

    def add_node(self, node_id: UUID) -> None:
        """
        Add a node to the layer with no connections.

        Args:
            node_id: Node to add
        """
        if node_id not in self._connections:
            self._connections[node_id] = []

    def get_neighbors(self, node_id: UUID) -> list[UUID]:
        """
        Get all neighbors of a node.

        Args:
            node_id: Node ID

        Returns:
            List of neighbor node IDs
        """
        return self._connections.get(node_id, [])

    def add_bidirectional_link(self, node1: UUID, node2: UUID) -> None:
        """
        Add a bidirectional connection between two nodes.

        Args:
            node1: First node
            node2: Second node
        """
        if node1 not in self._connections:
            self._connections[node1] = []
        if node2 not in self._connections:
            self._connections[node2] = []

        if node2 not in self._connections[node1]:
            self._connections[node1].append(node2)
        if node1 not in self._connections[node2]:
            self._connections[node2].append(node1)

    def set_neighbors(self, node_id: UUID, neighbors: list[UUID]) -> None:
        """
        Set the complete neighbor list for a node.

        Args:
            node_id: Node ID
            neighbors: New list of neighbors
        """
        self._connections[node_id] = neighbors

    def remove_node(self, node_id: UUID) -> None:
        """
        Remove a node and all its connections.

        Args:
            node_id: Node to remove
        """
        # Remove outgoing connections
        if node_id in self._connections:
            # Remove incoming connections from neighbors
            for neighbor_id in self._connections[node_id]:
                if neighbor_id in self._connections:
                    self._connections[neighbor_id] = [
                        nid for nid in self._connections[neighbor_id] if nid != node_id
                    ]
            del self._connections[node_id]

        # Remove any remaining incoming connections
        for nid in self._connections:
            self._connections[nid] = [n for n in self._connections[nid] if n != node_id]

    def contains(self, node_id: UUID) -> bool:
        """
        Check if a node exists in this layer.

        Args:
            node_id: Node to check

        Returns:
            True if node exists
        """
        return node_id in self._connections

    def clear(self) -> None:
        """Remove all nodes and connections."""
        self._connections.clear()


class HNSWIndex(VectorIndex):
    """
    HNSW (Hierarchical Navigable Small World) index.

    Uses a multi-layer graph structure where each layer is a navigable small world graph.
    Provides excellent search performance with logarithmic complexity.

    The algorithm maintains multiple layers of proximity graphs, with the top layers
    being sparser for long-range navigation and bottom layers denser for local search.
    """

    def __init__(
        self,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
    ):
        """
        Initialize HNSW index.

        Args:
            M: Maximum number of bi-directional links per node (higher = better recall, more memory)
            ef_construction: Size of dynamic candidate list during construction (higher = better quality)
            ef_search: Size of dynamic candidate list during search (higher = better recall)
            distance_metric: Distance metric to use
        """
        super().__init__(distance_metric)
        self.M = M
        self.M_max = M
        self.M_max_0 = M * 2  # Layer 0 has more connections
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.ml = 1.0 / np.log(2.0)  # Normalization factor for level assignment

        # Graph structure: list of layers
        self._layers: list[GraphLayer] = [GraphLayer()]

        # Chunk storage
        self._chunks: dict[UUID, Chunk] = {}

        # Entry point for search
        self._entry_point: UUID | None = None
        self._entry_point_layer: int = 0

        self._lock = ReadWriteLock()
        self._distance_fn = get_distance_function(distance_metric)

    def _select_level(self) -> int:
        """
        Select a random level for a new node using exponential decay.

        Returns:
            Layer number (0 is bottom layer)
        """
        rng = np.random.default_rng()
        return int(-np.log(rng.uniform(0, 1)) * self.ml)

    def _get_distance(self, chunk1: Chunk, chunk2: Chunk) -> float:
        """
        Compute distance between two chunks.

        Args:
            chunk1: First chunk
            chunk2: Second chunk

        Returns:
            Distance value
        """
        return self._distance_fn(chunk1.embedding, chunk2.embedding)

    def _search_layer(
        self,
        query_chunk: Chunk,
        entry_points: list[UUID],
        num_closest: int,
        layer: int,
    ) -> list[tuple[float, UUID]]:
        """
        Search for nearest neighbors in a specific layer.

        Args:
            query_chunk: Query chunk
            entry_points: List of entry point node IDs
            num_closest: Number of closest nodes to return
            layer: Layer to search

        Returns:
            List of (distance, node_id) tuples
        """
        visited: set[UUID] = set()
        candidates: list[tuple[float, UUID]] = []
        w: list[tuple[float, UUID]] = []

        # Initialize with entry points
        for ep_id in entry_points:
            if ep_id not in self._chunks:
                continue
            ep_chunk = self._chunks[ep_id]
            dist = self._get_distance(query_chunk, ep_chunk)
            heapq.heappush(candidates, (-dist, ep_id))  # Max heap (negative distance)
            heapq.heappush(w, (dist, ep_id))  # Min heap
            visited.add(ep_id)

        while candidates:
            # Get closest candidate
            current_dist, current_id = heapq.heappop(candidates)
            current_dist = -current_dist  # Convert back to positive

            # If this is farther than the worst result, we're done
            if current_dist > w[0][0] and len(w) >= num_closest:
                break

            # Explore neighbors
            if layer < len(self._layers):
                for neighbor_id in self._layers[layer].get_neighbors(current_id):
                    if neighbor_id not in visited and neighbor_id in self._chunks:
                        visited.add(neighbor_id)
                        neighbor_chunk = self._chunks[neighbor_id]
                        dist = self._get_distance(query_chunk, neighbor_chunk)

                        # If better than worst result, add to candidates
                        if dist < w[0][0] or len(w) < num_closest:
                            heapq.heappush(candidates, (-dist, neighbor_id))
                            heapq.heappush(w, (dist, neighbor_id))

                            # Keep only num_closest best results
                            if len(w) > num_closest:
                                heapq.heappop(w)

        return w

    def _get_neighbors(
        self, candidates: list[tuple[float, UUID]], M: int
    ) -> list[UUID]:
        """
        Select M neighbors using a heuristic (simple selection of closest).

        Args:
            candidates: List of (distance, node_id) candidates
            M: Number of neighbors to select

        Returns:
            List of selected neighbor node IDs
        """
        # Sort by distance and take M closest
        sorted_candidates = sorted(candidates, key=lambda x: x[0])
        return [node_id for _, node_id in sorted_candidates[:M]]

    def add(self, chunk: Chunk) -> None:
        """
        Add a chunk to the index.

        Args:
            chunk: Chunk to add

        Raises:
            ValueError: If embedding dimension doesn't match
        """
        self._validate_dimension(chunk.embedding)

        with self._lock.write():
            if chunk.id in self._chunks:
                return  # Already exists

            self._chunks[chunk.id] = chunk

            # Select layer for new element
            level = self._select_level()

            # Ensure we have enough layers
            while len(self._layers) <= level:
                self._layers.append(GraphLayer())

            # Initialize node in all layers from 0 to level
            for lc in range(level + 1):
                self._layers[lc].add_node(chunk.id)

            # If this is the first element, set as entry point
            if self._entry_point is None:
                self._entry_point = chunk.id
                self._entry_point_layer = level
                return

            # Search for nearest neighbors
            nearest = [self._entry_point]

            # Search from top layer to target layer
            for lc in range(self._entry_point_layer, level, -1):
                nearest = self._search_layer(chunk, nearest, 1, min(lc, len(self._layers) - 1))
                nearest = [node_id for _, node_id in nearest]

            # Insert into all layers from level down to 0
            for lc in range(level, -1, -1):
                candidates = self._search_layer(
                    chunk, nearest, self.ef_construction, lc
                )

                # Select M neighbors
                M = self.M_max_0 if lc == 0 else self.M_max
                neighbors = self._get_neighbors(candidates, M)

                # Add bidirectional links
                for neighbor_id in neighbors:
                    self._layers[lc].add_bidirectional_link(chunk.id, neighbor_id)

                    # Prune neighbor's connections if necessary
                    M_max = self.M_max_0 if lc == 0 else self.M_max
                    neighbor_connections = self._layers[lc].get_neighbors(neighbor_id)

                    if len(neighbor_connections) > M_max:
                        # Recompute distances and prune
                        neighbor_chunk = self._chunks[neighbor_id]
                        connections_with_distances = [
                            (
                                self._get_distance(
                                    neighbor_chunk, self._chunks[conn_id]
                                ),
                                conn_id,
                            )
                            for conn_id in neighbor_connections
                            if conn_id in self._chunks
                        ]
                        pruned_neighbors = self._get_neighbors(
                            connections_with_distances, M_max
                        )
                        self._layers[lc].set_neighbors(neighbor_id, pruned_neighbors)

                nearest = neighbors

            # Update entry point if necessary
            if level > self._entry_point_layer:
                self._entry_point = chunk.id
                self._entry_point_layer = level

    def add_batch(self, chunks: list[Chunk]) -> None:
        """
        Add multiple chunks at once.

        Args:
            chunks: List of chunks to add

        Raises:
            ValueError: If any embedding dimension doesn't match
        """
        for chunk in chunks:
            self._validate_dimension(chunk.embedding)

        # Call add() for each chunk (add() handles its own locking)
        for chunk in chunks:
            self.add(chunk)

    def search(
        self,
        query_embedding: list[float],
        k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """
        Search for k nearest neighbors.

        Args:
            query_embedding: Query vector
            k: Number of results to return
            metadata_filter: Optional metadata filters

        Returns:
            List of (chunk, distance) tuples sorted by distance

        Raises:
            ValueError: If query dimension doesn't match or index is empty
        """
        self._validate_dimension(query_embedding)

        with self._lock.read():
            if self._entry_point is None or len(self._chunks) == 0:
                return []

            # Create a temporary chunk for the query
            query_chunk = Chunk(
                id=UUID(int=0),
                text="<query>",
                embedding=query_embedding,
                document_id=UUID(int=0),
            )

            # Search from top layer to layer 0
            nearest = [self._entry_point]
            for lc in range(self._entry_point_layer, 0, -1):
                nearest = self._search_layer(query_chunk, nearest, 1, min(lc, len(self._layers) - 1))
                nearest = [node_id for _, node_id in nearest]

            # Search at layer 0 with ef_search
            candidates = self._search_layer(query_chunk, nearest, max(self.ef_search, k), 0)

            # Filter by metadata and convert to results
            results: list[tuple[float, Chunk]] = []
            for dist, node_id in candidates:
                if node_id in self._chunks:
                    chunk = self._chunks[node_id]
                    if self._matches_filter(chunk, metadata_filter):
                        results.append((dist, chunk))

            # Sort by distance and return top k
            results.sort(key=lambda x: x[0])
            return [(chunk, dist) for dist, chunk in results[:k]]

    def remove(self, chunk_id: UUID) -> bool:
        """
        Remove a chunk from the index.

        Args:
            chunk_id: ID of chunk to remove

        Returns:
            True if removed, False if not found
        """
        with self._lock.write():
            if chunk_id not in self._chunks:
                return False

            # Remove from all layers
            for layer in self._layers:
                layer.remove_node(chunk_id)

            # Remove chunk
            del self._chunks[chunk_id]

            # Update entry point if necessary
            if self._entry_point == chunk_id:
                if self._chunks:
                    # Set a new entry point (first available chunk)
                    self._entry_point = next(iter(self._chunks.keys()))
                    # Find its layer
                    self._entry_point_layer = 0
                    for lc, layer in enumerate(self._layers):
                        if layer.contains(self._entry_point):
                            self._entry_point_layer = lc
                else:
                    self._entry_point = None
                    self._entry_point_layer = 0

            return True

    def clear(self) -> None:
        """Remove all chunks from the index."""
        with self._lock.write():
            self._layers = [GraphLayer()]
            self._chunks.clear()
            self._entry_point = None
            self._entry_point_layer = 0
            self._dimension = None

    def size(self) -> int:
        """
        Get number of chunks in index.

        Returns:
            Number of chunks
        """
        with self._lock.read():
            return len(self._chunks)

    def __getstate__(self) -> dict:
        """
        Get state for pickling.

        Excludes the lock since it can't be pickled.
        """
        state = self.__dict__.copy()
        # Remove the lock - it will be recreated when unpickling
        state.pop('_lock', None)
        return state

    def __setstate__(self, state: dict) -> None:
        """
        Set state for unpickling.

        Recreates the lock.
        """
        self.__dict__.update(state)
        # Recreate the lock
        self._lock = ReadWriteLock()
