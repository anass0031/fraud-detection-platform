from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings, read from environment variables (set by docker compose from .env)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Fraud Detection API"
    version: str = "0.1.0"
    environment: str = "development"
    database_url: str = "postgresql+psycopg://fraud:fraud@localhost:5433/fraud"
    api_v1_prefix: str = "/api/v1"


@lru_cache
def get_settings() -> Settings:
    return Settings()
