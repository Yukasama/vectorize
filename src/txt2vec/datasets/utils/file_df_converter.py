"""Convert file to DataFrame."""

import tempfile
import uuid
from pathlib import Path

import aiofiles
import pandas as pd
from fastapi import UploadFile

from txt2vec.config import settings
from txt2vec.config.errors import ErrorNames

from ..exceptions import (
    EmptyFileError,
    FileTooLargeError,
    InvalidCSVFormatError,
    UnsupportedFormatError,
)
from ..file_format import FileFormat
from .file_loaders import load_file

__all__ = ["convert_file_to_df"]


_CHUNK_SIZE = 1_048_576
_FORMULA_PATTERNS = [b"=cmd", b"=sum", b"@", b"=importxml"]


async def convert_file_to_df(
    file: UploadFile, ext: str, sheet_index: int
) -> pd.DataFrame:
    """Convert uploaded file to pandas DataFrame.

    Securely processes files into DataFrames, checking for formula injection
    and malicious content across supported formats (CSV, JSON, Excel, etc).

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
        InvalidCSVFormatError: If the file contains malicious content like formulas.
    """
    first = await file.read(1)
    if not first:
        raise EmptyFileError
    await file.seek(0)

    try:
        file_format = FileFormat(ext)
    except ValueError as e:
        raise UnsupportedFormatError(ext) from e

    # Check for malicious content in the file header
    header = await file.read(_CHUNK_SIZE)
    header_lower = header.lower()

    # Check for null bytes (except in Excel files which may contain them)
    if b"\x00" in header and file_format != FileFormat.EXCEL:
        await file.seek(0)
        raise EmptyFileError

    # Check for formula injection patterns
    for pattern in _FORMULA_PATTERNS:
        if pattern in header_lower and file_format != FileFormat.EXCEL:
            await file.seek(0)
            raise InvalidCSVFormatError(ErrorNames.DETECT_MALICIOUS_CONTENT)

    await file.seek(0)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / uuid.uuid4().hex
        size = 0
        async with aiofiles.open(tmp_path, "wb") as tmp_file:
            while chunk := await file.read(_CHUNK_SIZE):
                size += len(chunk)
                if size > settings.dataset_max_upload_size:
                    raise FileTooLargeError(size)
                await tmp_file.write(chunk)
        await file.seek(0)

        try:
            df = load_file[file_format](tmp_path, sheet_index)
        except Exception as e:
            raise InvalidCSVFormatError from e

    if not isinstance(df, pd.DataFrame):
        raise InvalidCSVFormatError

    if df.empty:
        raise EmptyFileError

    return df
