"""Runtime settings for backend services.

Settings are loaded from environment variables or `.env`, keeping local
PostGIS connection details out of route handlers.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://geospatial:change_me_local_password@localhost:5432/landcover"
    aoi_path: str = "configs/aoi.geojson"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    """Cache settings so every request does not re-read environment files."""
    return Settings()
