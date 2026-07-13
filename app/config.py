"""Central application settings.

One validated object instead of os.environ reads scattered across modules.
Values come from the environment or a .env file in the working directory.

Two deliberate exceptions still read the environment directly:
- app/security.py — auth keys and the webhook secret are re-read per
  request so they can be rotated (and tests can monkeypatch them) without
  a restart.
- OPENAI_API_KEY / LANGSMITH_* — consumed by the OpenAI/LangSmith SDKs
  themselves; `load_dotenv()` in app.main exports the .env file so those
  SDKs see it.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    returns_db: str = "returns.db"
    returns_graph_db: str = "returns_graph.db"
    policy_catalog_csv: str | None = None
    policy_chunks_pkl: str | None = None
    label_expiry_days: int = 21
    board_confidence_threshold: float = 0.75
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
