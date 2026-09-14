
"""
Application Configuration.

Loads application settings from environment variables
and the .env file.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    # =========================================================
    # APPLICATION
    # =========================================================

    app_name: str = Field(
        default="AI Research Assistant"
    )

    app_version: str = Field(
        default="1.0.0"
    )

    environment: str = Field(
        default="development"
    )

    debug: bool = Field(
        default=True
    )

    # =========================================================
    # API
    # =========================================================

    api_v1_prefix: str = Field(
        default="/api/v1"
    )

    # =========================================================
    # DATABASE
    # =========================================================

    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/ai_research_assistant"
    )

    # =========================================================
    # DOCUMENT STORAGE
    # =========================================================

    upload_dir: str = Field(
        default="uploads"
    )

    max_upload_size_mb: int = Field(
        default=50
    )

    # =========================================================
    # TEXT CHUNKING
    # =========================================================

    chunk_size: int = Field(
        default=800
    )

    chunk_overlap: int = Field(
        default=100
    )

    # =========================================================
    # EMBEDDINGS
    # =========================================================

    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5"
    )

    embedding_dimension: int = Field(
        default=384
    )

    # =========================================================
    # QDRANT
    # =========================================================

    qdrant_host: str = Field(
        default="localhost"
    )

    qdrant_port: int = Field(
        default=6333
    )

    qdrant_collection: str = Field(
        default="document_chunks"
    )

    # =========================================================
    # LOCAL LLM
    # =========================================================

    llm_model: str = Field(
        default="qwen2.5:3b"
    )

    # =========================================================
    # PYDANTIC SETTINGS
    # =========================================================

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Return a cached application settings object.

    Caching ensures that the application uses one
    consistent settings instance.
    """

    return Settings()
