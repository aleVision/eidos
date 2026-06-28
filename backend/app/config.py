"""Application configuration.

All settings live in one place so the rest of the app never hard-codes paths,
URLs, or magic constants. Values can be overridden with environment variables
(e.g. EIDOS_DATABASE_URL) or a local .env file.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# The backend/ directory, regardless of where the app is launched from
BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EIDOS_", env_file=".env")

    app_name: str = "Eidos"
    app_version: str = "0.0.0"

    # SQLite lives in backedn/data/. The directory is created on startup
    database_url: str = f"sqlite:///{DATA_DIR / 'eidos.db'}"

    # The frontend dev server. Only these origins may call API (CORS)
    cors_origins: list[str] = ["http://127.0.0.1:5173", "http://localhost:5173"]


settings = Settings()
