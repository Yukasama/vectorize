"""Synthesis service layer."""

from uuid import UUID

from fastapi import HTTPException, status
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.datasets.exceptions import DatasetNotFoundError
from vectorize.datasets.repository import get_dataset_db

__all__ = [
    "validate_existing_dataset",
    "validate_upload_request",
]


async def validate_existing_dataset(
    db: AsyncSession,
    dataset_id_str: str,
) -> UUID:
    """Validate and convert dataset ID string to UUID.

    Args:
        db: Database session
        dataset_id_str: String representation of dataset UUID

    Returns:
        Valid dataset UUID

    Raises:
        HTTPException: If dataset ID is invalid or not found
    """
    try:
        dataset_uuid = UUID(dataset_id_str)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid UUID format: {dataset_id_str}",
        ) from e

    try:
        existing_dataset = await get_dataset_db(db, dataset_uuid)
        if not existing_dataset:
            raise DatasetNotFoundError(str(dataset_uuid))
    except DatasetNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID {dataset_uuid} not found.",
        ) from e

    return dataset_uuid


def validate_upload_request(
    files: list | None, existing_dataset_id: str | None
) -> None:
    """Validate that either files or dataset ID is provided.

    Args:
        files: List of uploaded files
        existing_dataset_id: Optional existing dataset ID

    Raises:
        HTTPException: If neither files nor dataset ID provided
    """
    if not files and existing_dataset_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either files or existing dataset ID must be provided.",
        )
