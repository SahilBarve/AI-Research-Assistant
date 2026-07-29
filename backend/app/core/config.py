from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Centralized configuration for the application.
    """

    app_name: str = Field(default="AI Research Assistant")
    app_version: str = Field(default="1.0.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=True)

    api_v1_prefix: str = Field(default="/api/v1")

    upload_dir: str = Field(default="uploads")
    max_upload_size_mb: int = Field(default=50)

    model_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    case_sensitive=False,
)
@lru_cache
def get_settings() -> Settings:
    """
    Return a cached Settings instance.
    """
    return Settings()