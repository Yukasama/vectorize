"""Dataset repository."""

from collections.abc import Sequence
from uuid import UUID

from loguru import logger
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.common.exceptions import VersionMismatchError

from .exceptions import DatasetNotFoundError
from .models import Dataset, DatasetUpdate

__all__ = [
    "get_dataset_db",
    "get_datasets_db",
    "update_dataset_db",
    "upload_dataset_db",
]


async def get_datasets_db(db: AsyncSession) -> Sequence[Dataset]:
    """Retrieve all datasets from the database.

    Args:
        db: Database session instance.

    Returns:
        A list of all Dataset objects in the database.
    """
    statement = select(Dataset)
    result = await db.exec(statement)
    datasets = result.all()

    logger.debug("Retrieved {} datasets from database", len(datasets))
    return datasets


async def get_dataset_db(db: AsyncSession, dataset_id: UUID) -> Dataset:
    """Retrieve a dataset by its ID.

    Args:
        db: Database session instance.
        dataset_id: The UUID of the dataset to retrieve.

    Returns:
        Dataset: The dataset object corresponding to the given ID.

    Raises:
        DatasetNotFoundError: If the dataset is not found.
    """
    statement = select(Dataset).where(Dataset.id == dataset_id)
    result = await db.exec(statement)
    dataset = result.first()

    if dataset is None:
        raise DatasetNotFoundError(dataset_id)

    return dataset


async def upload_dataset_db(db: AsyncSession, dataset: Dataset) -> UUID:
    """Save a new dataset to the database.

    Args:
        db: Database session instance.
        dataset: The dataset object to save.

    Returns:
        UUID: The UUID of the saved dataset.
    """
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)
    logger.debug("Dataset saved to DB", dataset=dataset)
    return dataset.id


async def update_dataset_db(
    db: AsyncSession,
    dataset_id: UUID,
    update_data: DatasetUpdate,
    expected_version: int,
) -> Dataset:
    """Update an existing dataset.

    Args:
        db: Database session instance.
        dataset_id: The UUID of the dataset to update.
        update_data: Dictionary containing the fields to update.
        expected_version: The expected version of the dataset for optimistic locking.

    Returns:
        Dataset: The updated dataset object.

    Raises:
        DatasetNotFoundError: If the dataset is not found.
    """
    result = await db.exec(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.first()

    if dataset is None:
        raise DatasetNotFoundError(dataset_id)

    if dataset.version != expected_version:
        raise VersionMismatchError(dataset_id, dataset.version)

    for field, value in update_data.model_dump(exclude_unset=True).items():
        setattr(dataset, field, value)

    dataset.version += 1

    await db.commit()
    await db.refresh(dataset)

    logger.debug("Dataset updated", dataset=dataset)
    return dataset


async def delete_dataset_db(db: AsyncSession, dataset_id: UUID) -> None:
    """Delete a dataset from the database.

    Args:
        db: Database session instance.
        dataset_id: The UUID of the dataset to delete.

    Raises:
        DatasetNotFoundError: If the dataset is not found.
    """
    statement = select(Dataset).where(Dataset.id == dataset_id)
    result = await db.exec(statement)
    dataset = result.first()

    if dataset is None:
        raise DatasetNotFoundError(dataset_id)

    await db.delete(dataset)
    await db.commit()
    logger.debug("Dataset deleted", dataset=dataset)
