"""
Application configuration.

This module centralizes all configurable settings for the
AI Research Assistant. Values are loaded from environment
variables or fall back to sensible defaults.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized configuration for the application.
    """

    # ---------------------------------------------------------
    # Application
    # ---------------------------------------------------------

    app_name: str = Field(default="AI Research Assistant")
    app_version: str = Field(default="1.0.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=True)

    # ---------------------------------------------------------
    # API
    # ---------------------------------------------------------

    api_v1_prefix: str = Field(default="/api/v1")

    # ---------------------------------------------------------
    # File Upload
    # ---------------------------------------------------------

    upload_dir: str = Field(default="uploads")
    max_upload_size_mb: int = Field(default=50)

    # ---------------------------------------------------------
    # Document Processing
    # ---------------------------------------------------------

    chunk_size: int = Field(default=800)
    chunk_overlap: int = Field(default=100)

    # ---------------------------------------------------------
    # Embeddings
    # ---------------------------------------------------------

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ---------------------------------------------------------
    # Vector Database
    # ---------------------------------------------------------

    qdrant_path: str = Field(default="qdrant_storage")

    # ---------------------------------------------------------
    # Environment
    # ---------------------------------------------------------

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.

    The configuration is loaded only once and reused
    throughout the application's lifetime.
    """
    return Settings()