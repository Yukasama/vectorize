"""Dataset service."""

import os
import string
import tempfile
import uuid
from pathlib import Path
from typing import Any, Final

import aiofiles
import pandas as pd
from fastapi import UploadFile
from loguru import logger

from txt2vec.config.config import (
    allowed_extensions,
    dataset_upload_dir,
    max_filename_length,
    max_upload_size,
)
from txt2vec.datasets.classification import Classification
from txt2vec.datasets.exceptions import (
    EmptyCSVError,
    FileTooLargeError,
    InvalidCSVFormatError,
    InvalidFileError,
    UnsupportedFormatError,
)
from txt2vec.datasets.file_format import FileFormat
from txt2vec.datasets.file_loaders import FILE_LOADERS
from txt2vec.datasets.models import Dataset
from txt2vec.datasets.repository import save_dataset

# -----------------------------------------------------------------------------
# Config ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

_ALLOWED_CHARS: Final[set[str]] = set(string.ascii_letters + string.digits + "_-")
CHUNK_SIZE = 1_048_576
MINIMUM_COLS = 2
MAXIMUM_COLS = 3


async def upload_file(file: UploadFile, sheet_name: int) -> dict[str, Any]:
    """Stream upload, parse to DataFrame, save as CSV, and return metadata.

    :param file: FastAPI ``UploadFile`` instance provided by the client.
    :param sheet_name: Sheet index to read when the file is an Excel workbook.
    :returns: Dictionary with ``filename``, ``rows``, ``columns``, and ``dataset_type``.
    :raises InvalidFileError: If filename is missing or the upload exceeds size limits.
    :raises UnsupportedFormatError: When the file extension is not supported.
    :raises EmptyCSVError: If the parsed DataFrame contains no rows.
    :raises InvalidCSVFormatError: If the DataFrame lacks required columns.
    """
    if not file.filename:
        raise InvalidFileError("Missing filename.")

    safe_name = _sanitize_filename(file.filename)
    ext = Path(safe_name).suffix.lstrip(".")

    try:
        file_format = FileFormat(ext)
    except ValueError as e:
        raise UnsupportedFormatError(ext) from e

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / uuid.uuid4().hex
        size = 0
        async with aiofiles.open(tmp_path, "wb") as tmp_file:
            while chunk := await file.read(CHUNK_SIZE):
                size += len(chunk)
                if size > max_upload_size:
                    raise FileTooLargeError(size)
                await tmp_file.write(chunk)
        await file.seek(0)

        df = FILE_LOADERS[file_format](tmp_path, sheet_name)

    if df.empty:
        raise EmptyCSVError

    _escape_csv_formulas(df)
    classification = _classify_dataset(df)

    unique_name = f"{Path(safe_name).stem}_{pd.Timestamp.utcnow():%Y%m%d_%H%M%S%f}.csv"
    _save_dataframe(df, unique_name)

    dataset = Dataset(
        name=unique_name,
        classification=classification,
        rows=len(df),
    )

    dataset_id = await save_dataset(dataset)
    logger.debug("Dataset saved", dataset=dataset)
    return dataset_id


# -----------------------------------------------------------------------------
# Utility ---------------------------------------------------------------------
# -----------------------------------------------------------------------------


def _sanitize_filename(filename: str) -> str:
    """Return a filesystem safe filename trimmed to allowed chars and length.

    :param filename: Raw filename supplied by the client request.
    :returns: Sanitised filename that can safely be written to disk.
    """
    base = os.path.basename(filename)
    stem, ext = os.path.splitext(base)
    ext = ext.lower().lstrip(".")

    if ext not in allowed_extensions:
        ext = ""

    stem_sanitized = "".join(c if c in _ALLOWED_CHARS else "_" for c in stem)
    if not stem_sanitized:
        stem_sanitized = "_"

    if len(stem_sanitized) > max_filename_length:
        stem_sanitized = stem_sanitized[:max_filename_length]

    return f"{stem_sanitized}.{ext}" if ext else stem_sanitized


def _escape_csv_formulas(df: pd.DataFrame) -> None:
    """Prefix dangerous strings with `'` so spreadsheets don't evaluate formulas.

    :param df: DataFrame to mutate in place for CSV export.
    """

    def needs_escape(val: str) -> bool:
        return val.startswith(("=", "-", "+", "@"))

    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].map(
                lambda x: f"'{x}" if isinstance(x, str) and needs_escape(x) else x
            )


def _classify_dataset(df: pd.DataFrame) -> Classification:
    """Return ``Classification`` enum value inferred from DataFrame columns.

    :param df: Loaded dataset as a DataFrame.
    :returns: ``Classification`` indicating duples or triples.
    :raises InvalidCSVFormatError: When the column layout is unsupported.
    """
    cols = {c.lower() for c in df.columns}
    if {"id", "anchor", "positive", "negative"}.issubset(cols):
        return Classification.SENTENCE_TRIPLES
    if len(df.columns) == MINIMUM_COLS:
        return Classification.SENTENCE_DUPLES
    if len(df.columns) == MAXIMUM_COLS:
        return Classification.SENTENCE_TRIPLES
    raise InvalidCSVFormatError


def _save_dataframe(df: pd.DataFrame, filename: str) -> Path:
    """Persist DataFrame as CSV in ``upload_dir`` and return its path.

    :param df: DataFrame to write.
    :param filename: Target filename (already sanitised).
    :returns: Path pointing to the saved CSV file.
    """
    out_path = dataset_upload_dir / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
