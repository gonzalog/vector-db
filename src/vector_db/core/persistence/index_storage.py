"""Index storage using pickle files."""

import pickle
from pathlib import Path
from typing import Optional, Any


class IndexStorage:
    """Manager for storing and loading pickled index structures."""

    def __init__(self, indexes_dir: Path):
        """
        Initialize index storage.

        Args:
            indexes_dir: Directory to store .pkl files
        """
        self.indexes_dir = indexes_dir
        self.indexes_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, library_id: str) -> Path:
        """Get path to .pkl file for a library."""
        return self.indexes_dir / f"{library_id}.pkl"

    def save(self, library_id: str, index: Any) -> None:
        """
        Save index structure to .pkl file.

        Args:
            library_id: Library ID
            index: Index object (FlatIndex, LSHIndex, or HNSWIndex)
        """
        path = self._get_path(library_id)

        # Atomic write: write to temp file, then rename
        temp_path = path.with_suffix(".pkl.tmp")
        with open(temp_path, "wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
        temp_path.rename(path)

    def load(self, library_id: str) -> Optional[Any]:
        """
        Load index structure from .pkl file.

        Args:
            library_id: Library ID

        Returns:
            Index object, or None if file doesn't exist or is corrupted
        """
        path = self._get_path(library_id)

        if not path.exists():
            return None

        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError, AttributeError):
            # Corrupted file - return None so index can be rebuilt
            return None

    def delete(self, library_id: str) -> None:
        """
        Delete index file for a library.

        Args:
            library_id: Library ID
        """
        path = self._get_path(library_id)
        if path.exists():
            path.unlink()

    def exists(self, library_id: str) -> bool:
        """
        Check if index file exists for a library.

        Args:
            library_id: Library ID

        Returns:
            True if file exists, False otherwise
        """
        return self._get_path(library_id).exists()
