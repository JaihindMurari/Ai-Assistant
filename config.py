"""
config.py
---------
Centralized application configuration.

All environment variables are loaded once here via python-dotenv and exposed
as a single `settings` object so the rest of the codebase never touches
`os.environ` directly. This keeps configuration concerns out of business
logic and makes unit testing easier (you can monkeypatch `settings`).
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables from .env (if present) before reading them.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _get_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(key: str, default: int) -> int:
    val = os.getenv(key)
    try:
        return int(val) if val else default
    except ValueError:
        return default


def _get_float(key: str, default: float) -> float:
    val = os.getenv(key)
    try:
        return float(val) if val else default
    except ValueError:
        return default


@dataclass
class Settings:
    """Strongly-typed application settings, populated from environment variables."""

    # --- LLM provider ---
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai").lower())

    # --- OpenAI ---
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    # --- Gemini ---
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))

    # --- Embeddings ---
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

    # --- Vector store ---
    vector_db_dir: str = field(default_factory=lambda: os.getenv("VECTOR_DB_DIR", "vector_db"))
    collection_name: str = field(default_factory=lambda: os.getenv("COLLECTION_NAME", "documents"))

    # --- Splitting ---
    chunk_size: int = field(default_factory=lambda: _get_int("CHUNK_SIZE", 1000))
    chunk_overlap: int = field(default_factory=lambda: _get_int("CHUNK_OVERLAP", 200))

    # --- Retrieval / generation defaults ---
    default_top_k: int = field(default_factory=lambda: _get_int("DEFAULT_TOP_K", 4))
    default_temperature: float = field(default_factory=lambda: _get_float("DEFAULT_TEMPERATURE", 0.2))
    default_max_tokens: int = field(default_factory=lambda: _get_int("DEFAULT_MAX_TOKENS", 1024))

    # --- Paths ---
    upload_dir: str = field(default_factory=lambda: os.getenv("UPLOAD_DIR", "data/uploads"))

    # --- OCR ---
    tesseract_cmd: str = field(default_factory=lambda: os.getenv("TESSERACT_CMD", ""))

    def has_valid_api_key(self) -> bool:
        """Return True if the currently selected provider has a usable API key."""
        if self.llm_provider == "openai":
            return bool(self.openai_api_key and self.openai_api_key != "your_openai_api_key_here")
        if self.llm_provider == "gemini":
            return bool(self.google_api_key and self.google_api_key != "your_google_api_key_here")
        return False

    def ensure_dirs(self) -> None:
        """Create required directories if they do not already exist."""
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.vector_db_dir).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Singleton settings instance used across the app.
# ---------------------------------------------------------------------------
settings = Settings()
settings.ensure_dirs()


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once. Safe to call multiple times."""
    root = logging.getLogger()
    if root.handlers:
        return  # already configured
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


configure_logging()
