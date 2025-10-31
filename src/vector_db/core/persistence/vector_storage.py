"""Vector storage using NumPy .npy files."""

import numpy as np
from pathlib import Path
from typing import Optional


class VectorStorage:
    """Manager for storing and loading vectors as .npy files."""

    def __init__(self, vectors_dir: Path):
        """
        Initialize vector storage.

        Args:
            vectors_dir: Directory to store .npy files
        """
        self.vectors_dir = vectors_dir
        self.vectors_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, library_id: str) -> Path:
        """Get path to .npy file for a library."""
        return self.vectors_dir / f"{library_id}.npy"

    def save(self, library_id: str, vectors: np.ndarray) -> None:
        """
        Save vectors to .npy file.

        Args:
            library_id: Library ID
            vectors: NumPy array of shape (n_vectors, embedding_dim)
        """
        path = self._get_path(library_id)

        # Atomic write: write to temp file, then rename
        # Note: np.save() automatically adds .npy extension
        temp_path_without_ext = self.vectors_dir / f"{library_id}.tmp"
        np.save(str(temp_path_without_ext), vectors)  # Creates library_id.tmp.npy

        # Rename from library_id.tmp.npy to library_id.npy
        temp_path_with_ext = Path(str(temp_path_without_ext) + ".npy")
        temp_path_with_ext.rename(path)

    def load(self, library_id: str) -> Optional[np.ndarray]:
        """
        Load vectors from .npy file.

        Args:
            library_id: Library ID

        Returns:
            NumPy array of shape (n_vectors, embedding_dim), or None if file doesn't exist
        """
        path = self._get_path(library_id)

        if not path.exists():
            return None

        return np.load(str(path))

    def delete(self, library_id: str) -> None:
        """
        Delete vectors file for a library.

        Args:
            library_id: Library ID
        """
        path = self._get_path(library_id)
        if path.exists():
            path.unlink()

    def exists(self, library_id: str) -> bool:
        """
        Check if vectors file exists for a library.

        Args:
            library_id: Library ID

        Returns:
            True if file exists, False otherwise
        """
        return self._get_path(library_id).exists()
