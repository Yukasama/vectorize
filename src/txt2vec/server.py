"""ASGI server for the FastAPI application."""

import uvicorn

from txt2vec.config import settings

__all__ = ["run"]


def run() -> None:
    """Run the FastAPI application using Uvicorn."""
    uvicorn.run(
        "txt2vec:app",
        port=settings.port,
        reload=True if settings.app_env == "production" else settings.reload,
        reload_dirs=["src"],
        server_header=settings.server_header,
    )
