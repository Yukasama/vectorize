"""ASGI server for the FastAPI application."""

import uvicorn
from uvicorn.config import LOGGING_CONFIG

from vectorize.config import settings

__all__ = ["run"]


def run() -> None:
    """Run the FastAPI application using Uvicorn."""
    is_production = settings.app_env == "production"

    uvicorn.run(
        "vectorize:app",
        port=settings.port,
        reload=settings.reload,
        reload_dirs=["src/vectorize"],
        server_header=False,
        log_config=None if is_production else LOGGING_CONFIG,
        log_level=None if is_production else "INFO",
    )
