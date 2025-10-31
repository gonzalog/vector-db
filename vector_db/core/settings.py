"""Application settings and configuration."""

import os
from pathlib import Path


class Settings:
    """Application settings loaded from environment variables."""

    # Data directory for persistence
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "./data"))

    # Database file path
    @property
    def DATABASE_PATH(self) -> Path:
        return self.DATA_DIR / "vector_db.sqlite"

    # Vectors directory
    @property
    def VECTORS_DIR(self) -> Path:
        return self.DATA_DIR / "vectors"

    # Indexes directory
    @property
    def INDEXES_DIR(self) -> Path:
        return self.DATA_DIR / "indexes"


# Global settings instance
settings = Settings()
