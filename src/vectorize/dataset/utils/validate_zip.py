"""Zip archive validation."""

import io
import zipfile
from collections.abc import Sequence
from typing import Final

from fastapi import UploadFile
from loguru import logger

from vectorize.common.exceptions import InvalidFileError
from vectorize.config.config import settings

from ..exceptions import FileTooLargeError, TooManyFilesError

__all__ = ["_handle_zip_upload", "_validate_zip_file"]


_MAX_ZIP_RATIO: Final[float] = 100.0


async def _handle_zip_upload(
    zip_file: UploadFile,
) -> list[UploadFile]:
    """Process and validate each file in a ZIP archive.

    Extracts and validates all files from the uploaded ZIP archive, then processes
    each valid file individually through the dataset upload pipeline.

    Args:
        zip_file: The uploaded ZIP file to process

    Returns:
        list[UploadFile]: List of files extracted from the ZIP archive

    Raises:
        InvalidFileError: If the ZIP file is corrupt or contains invalid data
        TooManyFilesError: If the ZIP contains more than the allowed number of files
        FileTooLargeError: If the uncompressed size exceeds the maximum limit
    """
    logger.debug("Processing ZIP file {}", zip_file.filename)
    content = await zip_file.read()

    members = _validate_zip_file(content)
    files = []

    for name, raw in members:
        files.append(UploadFile(filename=name, file=io.BytesIO(raw)))

    return files


def _validate_zip_file(zip_bytes: bytes) -> Sequence[tuple[str, bytes]]:
    """Validate and extract files from a ZIP archive with security checks.

    Performs security validation on a ZIP archive to prevent potential attacks:
    - Limits the number of files to prevent resource exhaustion
    - Checks uncompressed size to prevent disk space attacks
    - Verifies compression ratios to prevent zip bombs
    - Filters out system files and empty files

    Args:
        zip_bytes: Raw bytes of the ZIP file to validate

    Returns:
        Sequence[tuple[str, bytes]]: List of tuples containing filename and file content

    Raises:
        TooManyFilesError: If the ZIP contains more than the maximum allowed files
        FileTooLargeError: If the uncompressed size exceeds the limit
        InvalidFileError: If the ZIP contains empty files, has suspicious compression
                         ratios, or is otherwise invalid
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]

        if len(members) > settings.dataset_max_zip_members:
            raise TooManyFilesError(len(members))

        total_uncompressed = sum(m.file_size for m in members)
        if total_uncompressed > settings.dataset_max_upload_size:
            raise FileTooLargeError(size=total_uncompressed)

        safe_files: list[tuple[str, bytes]] = []

        for m in members:
            if m.filename.startswith(("__", ".")):
                continue

            if m.file_size == 0:
                safe_files.append((m.filename, b""))
                continue

            ratio = m.compress_size / m.file_size
            if ratio > _MAX_ZIP_RATIO:
                raise InvalidFileError(f"{m.filename}: suspicious ratio {ratio:0.1f}")

            with zf.open(m) as f:
                data = f.read()
                if not data.strip():
                    raise InvalidFileError(f"{m.filename} is empty")
                safe_files.append((m.filename, data))

        if not safe_files:
            raise InvalidFileError("ZIP contains no valid files")
        return safe_files
