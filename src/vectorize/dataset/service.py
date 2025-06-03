"""Dataset service."""

from pathlib import Path
from uuid import UUID, uuid4

from fastapi import Request, UploadFile
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.config import settings
from vectorize.dataset.column_mapper import ColumnMapping
from vectorize.utils.etag_parser import parse_etag
from vectorize.utils.file_sanitizer import sanitize_filename

from .models import Dataset, DatasetAll, DatasetPublic, DatasetUpdate
from .repository import (
    delete_dataset_db,
    get_dataset_db,
    get_datasets_db,
    update_dataset_db,
    upload_dataset_db,
)
from .upload_options_model import DatasetUploadOptions
from .utils.csv_escaper import _escape_csv_formulas
from .utils.dataset_classifier import _classify_dataset
from .utils.dataset_fs import _delete_dataset_from_fs, _save_dataframe_to_fs
from .utils.file_df_converter import _convert_file_to_df

__all__ = [
    "get_dataset_svc",
    "get_datasets_svc",
    "update_dataset_svc",
    "upload_dataset_svc",
]


async def get_datasets_svc(db: AsyncSession) -> list[DatasetAll]:
    """Read all datasets from the database.

    This function retrieves all datasets from the database and returns them as a
    list of dictionaries. Each dictionary contains limited fields (name,
    file_name, classification, created_at) for each dataset.

    Returns:
        List of dictionaries representing datasets with limited fields.
    """
    datasets = await get_datasets_db(db)

    return [DatasetAll.model_validate(dataset) for dataset in datasets]


async def get_dataset_svc(
    db: AsyncSession, dataset_id: UUID
) -> tuple[DatasetPublic, int]:
    """Read a single dataset from the database.

    This function retrieves a dataset by its ID from the database and returns it
    as a dictionary. The dictionary contains all fields of the dataset.

    Args:
        db: Database session for persistence operations
        dataset_id: The UUID of the dataset to retrieve

    Returns:
        Dictionary representing the dataset with all fields.
    """
    dataset = await get_dataset_db(db, dataset_id)
    return DatasetPublic.model_validate(dataset), dataset.version


async def upload_dataset_svc(
    db: AsyncSession, file: UploadFile, options: DatasetUploadOptions | None = None
) -> UUID:
    """Stream upload, parse file to DataFrame, save as JSONL, and return dataset ID.

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
    safe_name, ext = sanitize_filename(file, list(settings.allowed_extensions))

    column_mapping: ColumnMapping | None = None
    if options:
        column_mapping = ColumnMapping(
            prompt=options.prompt_name,
            chosen=options.chosen_name,
            rejected=options.rejected_name,
        )

    raw_df = await _convert_file_to_df(file, ext, options.sheet_index if options else 0)
    escaped_df = _escape_csv_formulas(raw_df)
    df, classification = _classify_dataset(escaped_df, column_mapping)

    try:
        unique_name = f"{Path(safe_name).stem}_{uuid4()}.jsonl"
        file_path = _save_dataframe_to_fs(df, unique_name)

        dataset = Dataset(
            name=safe_name,
            file_name=unique_name,
            classification=classification,
            rows=len(df),
        )
        logger.debug("Dataset DTO created", dataset=dataset)
        dataset_id = await upload_dataset_db(db, dataset)
        logger.debug("Dataset saved", datasetId=dataset_id)
        return dataset_id
    except Exception:
        # Roll back file on DB failure
        if "file_path" in locals() and Path(file_path).exists():
            Path(file_path).unlink()
        raise


async def update_dataset_svc(
    db: AsyncSession,
    request: Request,
    dataset_id: UUID,
    dataset_update: DatasetUpdate,
) -> int:
    """Update a dataset in the database.

    This function updates an existing dataset in the database with the provided
    data. It returns the updated dataset.

    Args:
        db: Database session for persistence operations
        request: The HTTP request object
        dataset_id: The UUID of the dataset to update
        dataset_update: The updated dataset data

    Returns:
        The updated dataset.
    """
    expected_version = parse_etag(str(dataset_id), request)

    updated_dataset = await update_dataset_db(
        db, dataset_id, dataset_update, expected_version
    )

    return updated_dataset.version


async def delete_dataset_svc(db: AsyncSession, dataset_id: UUID) -> None:
    """Delete a dataset from the database.

    This function deletes a dataset by its ID from the database.

    Args:
        db: Database session for persistence operations
        dataset_id: The UUID of the dataset to delete

    Returns:
        None
    """
    file_name = await delete_dataset_db(db, dataset_id)
    if not file_name:
        logger.warning("Dataset not found for deletion", dataset_id=dataset_id)
        return

    _delete_dataset_from_fs(file_name)
    logger.debug("Dataset deleted", dataset_id=dataset_id)
