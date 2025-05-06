"""ZIP archive validation."""

import io
import zipfile
from collections.abc import Sequence
from typing import Final
from uuid import UUID

from fastapi import UploadFile
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from ..exceptions import (
    FileTooLargeError,
    InvalidFileError,
    TooManyFilesError,
)
from ..service import upload_file
from ..upload_options_model import DatasetUploadOptions

__all__ = ["handle_zip_upload", "validate_zip_file"]


_MAX_ZIP_MEMBERS: Final[int] = 200
_MAX_ZIP_UNCOMPRESSED: Final[int] = 500 * 2**20
_MAX_ZIP_RATIO: Final[float] = 100.0


async def handle_zip_upload(
    zip_file: UploadFile,
    db: AsyncSession,
    options: DatasetUploadOptions,
) -> list[UUID]:
    """Process and validate each file in a ZIP archive.

    Extracts and validates all files from the uploaded ZIP archive, then processes
    each valid file individually through the dataset upload pipeline.

    Args:
        zip_file: The uploaded ZIP file to process
        db: Database session for persistence operations
        options: Configuration options for dataset processing

    Returns:
        list[UUID]: List of dataset IDs created from the ZIP contents

    Raises:
        InvalidFileError: If the ZIP file is corrupt or contains invalid data
        TooManyFilesError: If the ZIP contains more than the allowed number of files
        FileTooLargeError: If the uncompressed size exceeds the maximum limit
    """
    logger.debug("Processing ZIP file {}", zip_file.filename)
    content = await zip_file.read()

    members = validate_zip_file(content)
    dataset_ids: list[UUID] = []

    for name, raw in members:
        temp = UploadFile(filename=name, file=io.BytesIO(raw))
        dataset_ids.append(await upload_file(db, temp, options))

    return dataset_ids


def validate_zip_file(zip_bytes: bytes) -> Sequence[tuple[str, bytes]]:
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
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            members = [m for m in zf.infolist() if not m.is_dir()]

            if len(members) > _MAX_ZIP_MEMBERS:
                raise TooManyFilesError(len(members))

            total_uncompressed = sum(m.file_size for m in members)
            if total_uncompressed > _MAX_ZIP_UNCOMPRESSED:
                raise FileTooLargeError(size=total_uncompressed)

            safe_files: list[tuple[str, bytes]] = []

            for m in members:
                if m.filename.startswith(("__", ".")):
                    continue

                if m.file_size == 0:
                    raise InvalidFileError(f"{m.filename} is empty")

                ratio = m.compress_size / m.file_size
                if ratio > _MAX_ZIP_RATIO:
                    raise InvalidFileError(
                        f"{m.filename}: suspicious ratio {ratio:0.1f}"
                    )

                with zf.open(m) as f:
                    data = f.read()
                    if not data.strip():
                        raise InvalidFileError(f"{m.filename} is empty")
                    safe_files.append((m.filename, data))

            if not safe_files:
                raise InvalidFileError("ZIP contains no valid files")
            return safe_files

    except zipfile.BadZipFile as exc:
        raise InvalidFileError("Bad ZIP file") from exc
