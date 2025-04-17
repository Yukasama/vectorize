"""Logger configuration module."""

from pathlib import Path
from typing import Final

from loguru import logger

from txt2vec.config.config import app_config

__all__ = ["config_logger"]

log_config = app_config.get("logging", {})
rotation = log_config.get("rotation", "1 MB")

LOG_DIR: Final = Path(log_config.get("log_dir"))
LOG_FILE: Final = LOG_DIR / log_config.get("log_file")


def config_logger() -> None:
    """Logger configuration."""
    logger.add(LOG_FILE, rotation=rotation)
