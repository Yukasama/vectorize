"""ASGI server for the FastAPI application."""

import uvicorn

from txt2vec.config import settings

__all__ = ["run"]


def run() -> None:
    """Run the FastAPI application using Uvicorn."""
    uvicorn.run(
        "txt2vec:app",
        port=settings.port,
        reload=settings.reload,
        reload_dirs=["src/txt2vec"],
        server_header=settings.server_header,
    )
