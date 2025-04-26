"""Convert file to DataFrame."""

import tempfile
import uuid
from pathlib import Path

import aiofiles
from fastapi import UploadFile

from txt2vec.config.config import (
    max_upload_size,
)
from txt2vec.datasets.exceptions import (
    EmptyFileError,
    FileTooLargeError,
    UnsupportedFormatError,
)
from txt2vec.datasets.file_format import FileFormat
from txt2vec.datasets.file_loaders import FILE_LOADERS

__all__ = ["convert_file_to_df"]

_chunk_size = 1_048_576


async def convert_file_to_df(file: UploadFile, ext: str, sheet_index: int) -> bool:
    """Convert uploaded file to pandas DataFrame.

    Args:
        file: FastAPI UploadFile object containing the uploaded file.
        ext: File extension (e.g., 'csv', 'json').
        sheet_index: Sheet index for Excel files (ignored for other formats).

    Returns:
        pandas.DataFrame: DataFrame containing the file contents.

    Raises:
        EmptyFileError: If the file is empty.
        UnsupportedFormatError: If the file extension is not supported.
        FileTooLargeError: If the file exceeds the maximum allowed size.
    """
    first = await file.read(1)
    if not first:
        raise EmptyFileError
    await file.seek(0)

    try:
        file_format = FileFormat(ext)
    except ValueError as e:
        raise UnsupportedFormatError(ext) from e

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / uuid.uuid4().hex
        size = 0
        async with aiofiles.open(tmp_path, "wb") as tmp_file:
            while chunk := await file.read(_chunk_size):
                size += len(chunk)
                if size > max_upload_size:
                    raise FileTooLargeError(size)
                await tmp_file.write(chunk)
        await file.seek(0)

        df = FILE_LOADERS[file_format](tmp_path, sheet_index)

    if df.empty:
        raise EmptyFileError

    return df
