from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PLACEHOLDER_SECRET = "change-me-generate-a-long-random-secret"


class Settings(BaseSettings):
    """App settings, read from environment variables (set by docker compose from .env)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Fraud Detection API"
    version: str = "0.1.0"
    environment: str = "development"
    database_url: str = "postgresql+psycopg://fraud:fraud@localhost:5433/fraud"
    api_v1_prefix: str = "/api/v1"

    # --- authentication ---
    jwt_secret: str  # required: no default, so a missing secret stops the app at startup
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 15
    refresh_token_days: int = 7
    bcrypt_rounds: int = 12
    # Secure cookies need HTTPS; local development runs on plain http.
    cookie_secure: bool = False

    @field_validator("jwt_secret")
    @classmethod
    def _strong_secret(cls, v: str) -> str:
        if v == PLACEHOLDER_SECRET or len(v) < 32:
            raise ValueError("JWT_SECRET must be set to a random value of at least 32 characters")
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
