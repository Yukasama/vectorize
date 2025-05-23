"""Sanitize filenames for file uploads."""

import string
from pathlib import Path
from typing import Final

from fastapi import UploadFile

from vectorize.config import settings
from vectorize.config.errors import ErrorNames
from vectorize.datasets.exceptions import InvalidFileError, UnsupportedFormatError

__all__ = ["sanitize_filename"]


_ALLOWED_CHARS: Final[set[str]] = set(string.ascii_letters + string.digits + "_-")


def sanitize_filename(
    file: UploadFile, allowed_extensions: list[str]
) -> tuple[str, str]:
    """Return a filesystem safe filename trimmed to allowed chars and length.

    Args:
        file: UploadFile object containing the file to be sanitized.
        allowed_extensions: List of permitted file extensions.

    Returns:
        str: Sanitized filename that can safely be written to disk.

    Raises:
        InvalidFileError: If the filename is missing or too long.
    """
    if not file.filename:
        raise InvalidFileError(ErrorNames.FILENAME_MISSING_ERROR)

    base = Path(file.filename).name
    stem = Path(base).stem
    ext = Path(base).suffix.lstrip(".").lower().lstrip(".")

    if not stem or len(stem) == 0 or not ext:
        raise InvalidFileError(ErrorNames.FILENAME_MISSING_ERROR)

    if not ext or ext not in allowed_extensions:
        raise UnsupportedFormatError(ext)

    stem_sanitized = "".join(c if c in _ALLOWED_CHARS else "_" for c in stem)
    if not stem_sanitized:
        stem_sanitized = "_"

    if len(stem_sanitized) > settings.max_filename_length:
        raise InvalidFileError(ErrorNames.FILENAME_TOO_LONG_ERROR)

    return f"{stem_sanitized}.{ext}" if ext else stem_sanitized, ext
