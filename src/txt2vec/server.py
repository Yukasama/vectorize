"""ASGI server for the FastAPI application."""

import uvicorn

from txt2vec.config.config import app_config

__all__ = ["run"]

server_config = app_config.get("server", {})


def run() -> None:
    """Run the FastAPI application using Uvicorn."""
    uvicorn.run(
        "txt2vec:app",
        port=server_config.get("port", 8000),
        reload=server_config.get("reload", False),
        reload_dirs=["src"],
        server_header=server_config.get("server_header", False),
    )
