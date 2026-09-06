from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ==========================================
    # Application
    # ==========================================

    app_name: str = Field(default="AI Research Assistant")
    app_version: str = Field(default="1.0.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=True)

    # ==========================================
    # API
    # ==========================================

    api_v1_prefix: str = Field(default="/api/v1")

    # ==========================================
    # File Uploads
    # ==========================================

    upload_dir: str = Field(default="uploads")
    max_upload_size_mb: int = Field(default=50)

    # ==========================================
    # Document Processing
    # ==========================================

    chunk_size: int = Field(default=800)
    chunk_overlap: int = Field(default=100)

    # ==========================================
    # Embeddings
    # ==========================================

    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5"
    )

    embedding_dimension: int = Field(default=384)

    # ==========================================
    # Qdrant
    # ==========================================

    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)

    qdrant_collection: str = Field(
        default="document_chunks"
    )

    # ==========================================
    # LLM
    # ==========================================

    llm_model: str = Field(
        default="qwen2.5:3b"
    )

    # ==========================================
    # Configuration
    # ==========================================

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()