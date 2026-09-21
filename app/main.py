"""FastAPI application entry point.

Run locally:
    uvicorn app.main:app --reload

The application object is ``app`` and is what production servers should
target (e.g. ``uvicorn app.main:app --host 0.0.0.0 --port 8000``).
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import STATIC_DIR, settings


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
)
logger = logging.getLogger("gridwise")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    docs_url=None,           # Replaced by our themed /docs route
    redoc_url=None,
    swagger_ui_parameters={
        "syntaxHighlight.theme": "obsidian",
        "defaultModelsExpandDepth": -1,
    },
)

# Serve static assets (CSS, JS, icons) — but the dashboard HTML is served
# directly by the routes module so it can refresh on every request.
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(router)


@app.on_event("startup")
async def _startup() -> None:
    logger.info(
        "GridWise LLM v%s starting — LLM configured=%s, models=%s",
        settings.APP_VERSION,
        settings.llm_configured,
        settings.candidate_models(),
    )


# ---------------------------------------------------------------------------
# Convenience: ``python -m app.main``
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", settings.PORT))
    uvicorn.run("app.main:app", host=settings.HOST, port=port, reload=False)
