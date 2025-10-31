"""Application configuration."""

from typing import Literal


class Settings:
    """Application settings."""

    PROJECT_NAME: str = "Vector DB API"
    VERSION: str = "0.1.0"
    API_V1_PREFIX: str = "/api/v1"

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS settings
    BACKEND_CORS_ORIGINS: list[str] = ["*"]

    # Environment
    ENVIRONMENT: Literal["development", "production"] = "development"


settings = Settings()
