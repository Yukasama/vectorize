"""Logger configuration module."""

from pathlib import Path
from typing import Final

from loguru import logger

__all__ = ["config_logger"]

LOG_DIR: Final = Path("log")
LOG_FILE: Final = LOG_DIR / "app.log"


def config_logger() -> None:
    """Logger configuration."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(LOG_FILE, rotation="1 MB")
