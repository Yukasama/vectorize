"""Utils for datasets."""

from typing import Final

BYTES_PER_KB: Final[int] = 1024
BYTES_PER_MB: Final[int] = 1024 * BYTES_PER_KB
BYTES_PER_GB: Final[int] = 1024 * BYTES_PER_MB


def format_file_size(size_in_bytes: int) -> str:
    """Format file size to be more human-readable.

    :param size_in_bytes: Size in bytes to format
    :return: Human-readable size string (e.g., "4.20 MB")
    """
    if size_in_bytes >= BYTES_PER_GB:
        return f"{size_in_bytes / BYTES_PER_GB:.2f} GB"
    if size_in_bytes >= BYTES_PER_MB:
        return f"{size_in_bytes / BYTES_PER_MB:.2f} MB"
    if size_in_bytes >= BYTES_PER_KB:
        return f"{size_in_bytes / BYTES_PER_KB:.2f} KB"
    return f"{size_in_bytes} bytes"
