"""ASGI server for the FastAPI application."""

import uvicorn

__all__ = ["run"]


def run() -> None:
    """Run the FastAPI application using Uvicorn."""
    uvicorn.run(
        "txt2vec:app",
        port=8000,
        reload=True,
        reload_dirs=["src"],
        server_header=False,
    )
