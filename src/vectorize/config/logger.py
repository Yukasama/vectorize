"""Logger configuration."""

import sys
from collections.abc import Mapping
from typing import Any

from loguru import logger
from loki_logger_handler.formatters.loguru_formatter import LoguruFormatter
from loki_logger_handler.loki_logger_handler import LokiLoggerHandler

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

    logger.add(
        sys.stdout,
        # format=_format_record,
        level=settings.log_level,
        colorize=True,
        enqueue=True,
    )

    loki_handler = LokiLoggerHandler(
        url="http://localhost:9999/loki/api/v1/push",
        labels={"application": "fastapi", "environment": "dev"},
        timeout=10,
        enable_structured_loki_metadata=True,
        default_formatter=LoguruFormatter(),
    )
    logger.add(loki_handler)


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
