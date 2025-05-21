"""Logger configuration."""

import sys
from collections.abc import Mapping
from typing import Any

from loguru import logger
from loki_handler import LokiHandler

from .config import settings

__all__ = ["config_logger"]


def config_logger() -> None:
    """Logger configuration."""
    logger.remove()

    is_production = settings.app_env == "production"

    logger.add(
        settings.log_path,
        rotation=settings.rotation,
        format=_format_record,
        enqueue=not is_production,
        backtrace=not is_production,
        diagnose=not is_production,
        compression="zip",
        colorize=False,
    )

    if not is_production:
        logger.add(
            sys.stdout,
            format=_format_record,
            level=settings.log_level,
            colorize=True,
            enqueue=True,
        )

    logger.add(
        LokiHandler(
            url="http://localhost:9999/loki/api/v1/push",
            tags={"application": "fastapi"},
            version="1",
        ),
        format="{message}",
        serialize=True,
        level=settings.log_level,
    )


def _format_record(record: Mapping[str, Any]) -> str:
    ts = record["time"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    line = (
        f"<green>{ts}</green> | "
        f"<level>{record['level']:<8}</level> | "
        f"<cyan>{record['name']}:{record['function']}:{record['line']}</cyan> - "
        f"{record['message']}"
    )

    if record["extra"]:
        extras = " | ".join(
            f"<yellow>{k}</yellow>=<cyan>{v}</cyan>" for k, v in record["extra"].items()
        )
        line += f" | {extras}"

    return line + "\n"
