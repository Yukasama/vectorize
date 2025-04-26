"""Format file size to be more human-readable."""

from typing import Final

__all__ = ["format_file_size"]

_bytes_per_kb: Final[int] = 1024
_bytes_per_mb: Final[int] = 1024 * _bytes_per_kb
_bytes_per_gb: Final[int] = 1024 * _bytes_per_mb


def format_file_size(size_in_bytes: int) -> str:
    """Format file size to be more human-readable.

    Args:
        size_in_bytes: Size in bytes to format

    Returns:
        Human-readable size string (e.g., "4.20 MB")
    """
    if size_in_bytes >= _bytes_per_gb:
        return f"{size_in_bytes / _bytes_per_gb:.2f} GB"
    if size_in_bytes >= _bytes_per_mb:
        return f"{size_in_bytes / _bytes_per_mb:.2f} MB"
    if size_in_bytes >= _bytes_per_kb:
        return f"{size_in_bytes / _bytes_per_kb:.2f} KB"
    return f"{size_in_bytes} bytes"
