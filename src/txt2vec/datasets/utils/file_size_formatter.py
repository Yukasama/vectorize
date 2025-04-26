"""Format file size to be more human-readable."""

from typing import Final

__all__ = ["format_file_size"]

_BYTES_PER_KB: Final[int] = 1024
_BYTES_PER_MB: Final[int] = 1024 * _BYTES_PER_KB
_BYTES_PER_GB: Final[int] = 1024 * _BYTES_PER_MB


def format_file_size(size_in_bytes: int) -> str:
    """Format file size to be more human-readable.

    Args:
        size_in_bytes: Size in bytes to format

    Returns:
        Human-readable size string (e.g., "4.20 MB")
    """
    if size_in_bytes >= _BYTES_PER_GB:
        return f"{size_in_bytes / _BYTES_PER_GB:.2f} GB"
    if size_in_bytes >= _BYTES_PER_MB:
        return f"{size_in_bytes / _BYTES_PER_MB:.2f} MB"
    if size_in_bytes >= _BYTES_PER_KB:
        return f"{size_in_bytes / _BYTES_PER_KB:.2f} KB"
    return f"{size_in_bytes} bytes"
