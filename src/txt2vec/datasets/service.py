"""Dataset service."""

import uuid
from pathlib import Path

import pandas as pd
from fastapi import UploadFile
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from txt2vec.config.config import allowed_extensions
from txt2vec.utils import sanitize_filename

from .classification import classify_dataset
from .column_mapper import ColumnMapping
from .exceptions import InvalidFileError
from .models import Dataset
from .repository import save_dataset
from .utils.csv_escaper import escape_csv_formulas
from .utils.file_df_converter import convert_file_to_df
from .utils.save_dataset import save_dataframe

__all__ = ["upload_file"]


async def upload_file(
    db: AsyncSession,
    file: UploadFile,
    column_mapping: ColumnMapping | None = None,
    sheet_index: int = 0,
) -> uuid.UUID:
    """Stream upload, parse file to DataFrame, save as CSV, and return dataset ID.

    Args:
        db: Database session for storing the dataset information.
        file: FastAPI UploadFile instance provided by the client.
        column_mapping: Optional mapping to rename standard columns in the dataset.
        sheet_index: Sheet index to read when the file is an Excel workbook.

    Returns:
        UUID of the created dataset record.

    Raises:
        InvalidFileError: If file, filename is missing or the upload exceeds size
        limits.
        UnsupportedFormatError: When the file extension is not supported.
        EmptyFileError: If the parsed DataFrame contains no rows.
        InvalidCSVFormatError: If the DataFrame lacks required columns.
        FileTooLargeError: If the uploaded file exceeds the maximum size limit.
    """
    if file is None:
        raise InvalidFileError("No file provided")

    safe_name = sanitize_filename(file, allowed_extensions)
    ext = Path(safe_name).suffix.lstrip(".")

    raw_df = await convert_file_to_df(file, ext, sheet_index)
    df, classification = classify_dataset(raw_df, column_mapping)
    escape_csv_formulas(df)

    unique_name = f"{Path(safe_name).stem}_{pd.Timestamp.utcnow():%Y%m%d_%H%M%S%f}.csv"
    save_dataframe(df, unique_name)

    dataset = Dataset(
        name=safe_name,
        file_name=unique_name,
        classification=classification,
        rows=len(df),
    )

    logger.debug("Dataset DTO created", dataset=dataset)
    dataset_id = await save_dataset(db, dataset)
    logger.debug("Dataset saved", datasetId=dataset_id)
    return dataset_id
