"""Runtime settings. Everything is overridable with EPOCH_* environment variables or a .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EPOCH_", env_file=".env", extra="ignore")

    home: Path = Path(".epoch")
    db_url: str | None = None  # default: sqlite in `home`; set postgresql+psycopg://... for pgvector
    redis_url: str | None = None  # enables cross-process event fan-out (worker -> API SSE)
    mlflow_uri: str | None = None  # mirror trials into MLflow when set
    otlp_endpoint: str | None = None

    anthropic_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY", "EPOCH_ANTHROPIC_API_KEY")
    )
    agent_model: str = "claude-sonnet-5"
    judge_model: str = "claude-haiku-4-5-20251001"
    triage_haiku_model: str = "claude-haiku-4-5-20251001"
    triage_sonnet_model: str = "claude-sonnet-5"
    offline: bool = False  # force heuristic agents even when a key is present

    hw_price_per_hour: float | None = None  # $/h of the serving hardware; auto-picked per device if unset
    device: str | None = None  # "cuda" | "cpu"; auto-detected if unset

    @property
    def llm_enabled(self) -> bool:
        return bool(self.anthropic_api_key) and not self.offline

    @property
    def resolved_db_url(self) -> str:
        if self.db_url:
            return self.db_url
        self.home.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{(self.home / 'epoch.db').resolve()}"


@lru_cache
def settings() -> Settings:
    return Settings()


@lru_cache
def pricing() -> dict:
    with open(CONFIG_DIR / "pricing.yaml") as f:
        return yaml.safe_load(f)
