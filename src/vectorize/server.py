"""ASGI server for the FastAPI application."""

import os
from pathlib import Path

import uvicorn

from vectorize.config import settings

__all__ = ["run"]


def run() -> None:
    """Run the FastAPI application using Uvicorn."""
    base_dir = Path(
        os.environ.get("CERT_DIR", Path(__file__).parent / "config" / "resources")
    )

    ssl_keyfile = Path(os.environ.get("SSL_KEYFILE", str(base_dir / "key.pem")))
    ssl_certfile = Path(os.environ.get("SSL_CERTFILE", str(base_dir / "cert.pem")))
    ssl_enabled = ssl_keyfile.exists() and ssl_certfile.exists()

    uvicorn.run(
        "vectorize:app",
        port=settings.port,
        reload=settings.reload,
        reload_dirs=["src/vectorize"],
        server_header=settings.server_header,
        log_config=None,
        log_level=None,
        ssl_keyfile=ssl_keyfile if ssl_enabled else None,
        ssl_certfile=ssl_certfile if ssl_enabled else None,
    )
