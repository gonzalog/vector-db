"""Application configuration."""

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application Info
    PROJECT_NAME: str = Field(default="Vector DB API", description="Project name")
    VERSION: str = Field(default="0.1.0", description="API version")
    API_V1_PREFIX: str = Field(default="/api/v1", description="API v1 prefix")

    # Server settings
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")

    # Environment
    ENVIRONMENT: Literal["development", "production"] = Field(
        default="development", description="Environment"
    )

    # CORS settings
    BACKEND_CORS_ORIGINS: list[str] | str = Field(
        default=["*"], description="CORS allowed origins"
    )

    # Third-party API Keys
    COHERE_API_KEY: str = Field(default="", description="Cohere API key")
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    ANTHROPIC_API_KEY: str = Field(default="", description="Anthropic API key")

    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    DEBUG: bool = Field(default=False, description="Debug mode")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    def __init__(self, **kwargs):
        """Initialize settings and parse CORS origins if provided as string."""
        super().__init__(**kwargs)

        # Parse CORS origins if provided as comma-separated string
        if isinstance(self.BACKEND_CORS_ORIGINS, str):
            self.BACKEND_CORS_ORIGINS = [
                origin.strip()
                for origin in self.BACKEND_CORS_ORIGINS.split(",")
                if origin.strip()
            ]


settings = Settings()
