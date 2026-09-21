"""Application configuration and environment variable management.

Centralizes all tunables, defaults, and runtime paths. Keeping this in one
place makes deployment configuration trivial and the rest of the codebase
free of ``os.getenv`` calls.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv


# Resolve project root regardless of where the app is launched from.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
STATIC_DIR: Path = PROJECT_ROOT / "static"
QUESTION_DIR: Path = PROJECT_ROOT / "Question"
SAMPLE_CASES_FILE: Path = (
    QUESTION_DIR / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"
)
FALLBACK_PRESETS_FILE: Path = PROJECT_ROOT / "static_presets.json"

# Load .env once at module import. Subsequent loads are no-ops.
load_dotenv(PROJECT_ROOT / ".env")


class Settings:
    """Application settings loaded from environment variables.

    Reads tunables lazily so test overrides can mutate ``os.environ`` before
    instantiation if needed.
    """

    # ---- Service metadata --------------------------------------------------
    APP_NAME: str = "GridWise LLM"
    APP_VERSION: str = "2.0.0"
    APP_DESCRIPTION: str = (
        "BUP CSE Fest 2026 — Smart Campus Energy Optimization Engine"
    )

    # ---- Server settings ---------------------------------------------------
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # ---- LLM settings ------------------------------------------------------
    DEFAULT_GEMINI_MODEL: str = "gemini-2.5-flash"
    DEFAULT_GEMINI_FALLBACKS: str = "gemini-2.0-flash,gemini-1.5-flash"
    LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "12"))

    # ---- Prompt tuning -----------------------------------------------------
    LLM_TEMPERATURE: float = 0.0

    # -----------------------------------------------------------------------
    # Helper accessors
    # -----------------------------------------------------------------------
    @property
    def gemini_api_key(self) -> Optional[str]:
        """Return the Gemini API key from environment, or ``None``."""
        return os.getenv("GEMINI_API_KEY") or None

    @property
    def llm_configured(self) -> bool:
        """Whether an LLM API key is present (any provider)."""
        return bool(self.gemini_api_key or os.getenv("OPENAI_API_KEY"))

    def candidate_models(self) -> List[str]:
        """Return an ordered, deduplicated list of LLM model names to try."""
        primary = os.getenv("GEMINI_MODEL", self.DEFAULT_GEMINI_MODEL).strip()
        fallbacks_raw = os.getenv(
            "GEMINI_FALLBACK_MODELS", self.DEFAULT_GEMINI_FALLBACKS
        )
        fallbacks = [m.strip() for m in fallbacks_raw.split(",") if m.strip()]
        seen: List[str] = []
        for name in [primary] + fallbacks:
            if name and name not in seen:
                seen.append(name)
        return seen


# Singleton-style accessor.  Imported as ``settings`` elsewhere.
settings = Settings()
