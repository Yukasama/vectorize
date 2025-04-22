"""ASGI server for the FastAPI application."""

import uvicorn

from txt2vec.config.config import port, reload, server_header

__all__ = ["run"]


def run() -> None:
    """Run the FastAPI application using Uvicorn."""
    uvicorn.run(
        "txt2vec:app",
        port=port,
        reload=reload,
        reload_dirs=["src"],
        server_header=server_header,
    )
