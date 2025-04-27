"""ASGI server for the FastAPI application."""

import uvicorn

from txt2vec.config.config import app_env, port, reload, server_header

__all__ = ["run"]


def run() -> None:
    """Run the FastAPI application using Uvicorn."""
    uvicorn.run(
        "txt2vec:app",
        port=port,
        reload=True if app_env == "production" else reload,
        reload_dirs=["src"],
        server_header=server_header,
    )
