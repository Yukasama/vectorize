"""Logger configuration module."""

import sys

from loguru import logger

from txt2vec.config.config import log_path, rotation

__all__ = ["config_logger"]


def format_record(record):
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


def config_logger() -> None:
    """Logger configuration."""
    logger.remove()

    logger.add(
        log_path,
        rotation=rotation,
        format=format_record,
        enqueue=True,
    )

    logger.add(
        sys.stdout, format=format_record, level="DEBUG", colorize=True, enqueue=True
    )
