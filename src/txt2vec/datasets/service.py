"""Dataset service."""

import uuid
from pathlib import Path

from fastapi import UploadFile
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from txt2vec.config.config import allowed_extensions
from txt2vec.datasets.upload_options_model import DatasetUploadOptions
from txt2vec.utils import sanitize_filename

from .classification import classify_dataset
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
    options: DatasetUploadOptions | None = None,
) -> uuid.UUID:
    """Stream upload, parse file to DataFrame, save as CSV, and return dataset ID.

    Args:
        db: Database session for storing the dataset information.
        file: FastAPI UploadFile instance provided by the client.
        options: DatasetUploadOptions instance containing column names and
            sheet index for Excel files.

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

    safe_name, ext = sanitize_filename(file, allowed_extensions)

    column_mapping = {
        "question": options.question_name,
        "positive": options.positive_name,
        "negative": options.negative_name,
    }

    raw_df = await convert_file_to_df(file, ext, options.sheet_index)
    escaped_df = escape_csv_formulas(raw_df)
    df, classification = classify_dataset(escaped_df, column_mapping)

    try:
        unique_name = f"{Path(safe_name).stem}_{uuid.uuid4()}.csv"
        file_path = save_dataframe(df, unique_name)

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
    except Exception:
        # Clean up the saved file if database operation failed
        if "file_path" in locals() and Path(file_path).exists():
            Path(file_path).unlink()
        raise
