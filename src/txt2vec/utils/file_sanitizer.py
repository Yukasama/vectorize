"""Sanitize filenames for file uploads."""

import string
from pathlib import Path
from typing import Final

from fastapi import UploadFile

from txt2vec.config.config import max_filename_length
from txt2vec.datasets.exceptions import InvalidFileError

__all__ = ["sanitize_filename"]

_allowed_chars: Final[set[str]] = set(string.ascii_letters + string.digits + "_-")


def sanitize_filename(file: UploadFile, allowed_extensions: list[str]) -> str:
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
        raise InvalidFileError("Missing filename.")

    base = Path.name(file.filename)
    stem = Path(base).stem
    ext = Path(base).suffix.lstrip(".")
    ext = ext.lower().lstrip(".")

    if ext not in allowed_extensions:
        ext = ""

    stem_sanitized = "".join(c if c in _allowed_chars else "_" for c in stem)
    if not stem_sanitized:
        stem_sanitized = "_"

    if len(stem_sanitized) > max_filename_length:
        raise InvalidFileError("Filename is too long.")

    return f"{stem_sanitized}.{ext}" if ext else stem_sanitized
