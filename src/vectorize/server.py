"""ASGI server for the FastAPI application."""

import uvicorn

from vectorize.config import settings

__all__ = ["run"]


def run() -> None:
    """Run the FastAPI application using Uvicorn."""
    uvicorn.run(
        "vectorize:app",
        port=settings.port,
        reload=settings.reload,
        reload_dirs=["src/vectorize"],
        server_header=settings.server_header,
    )
