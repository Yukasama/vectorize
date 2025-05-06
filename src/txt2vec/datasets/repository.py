"""Dataset repository."""

from uuid import UUID

from loguru import logger
from sqlmodel import select, update
from sqlmodel.ext.asyncio.session import AsyncSession

from .exceptions import DatasetNotFoundError
from .models import Dataset, DatasetUpdate

__all__ = ["get_all_datasets", "get_dataset", "save_dataset", "update_dataset"]


async def save_dataset(db: AsyncSession, dataset: Dataset) -> UUID:
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


async def get_dataset(db: AsyncSession, dataset_id: UUID) -> Dataset:
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
        raise DatasetNotFoundError(str(dataset_id))

    return dataset


async def get_all_datasets(db: AsyncSession) -> list[Dataset]:
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


async def update_dataset(
    db: AsyncSession, dataset_id: UUID, update_data: DatasetUpdate
) -> Dataset:
    """Update an existing dataset.

    Args:
        db: Database session instance.
        dataset_id: The UUID of the dataset to update.
        update_data: Dictionary containing the fields to update.

    Returns:
        Dataset: The updated dataset object.

    Raises:
        DatasetNotFoundError: If the dataset is not found.
    """
    statement = select(Dataset).where(Dataset.id == dataset_id)
    result = await db.exec(statement)
    dataset = result.first()

    if dataset is None:
        raise DatasetNotFoundError(str(dataset_id))

    update_data["version"] = dataset.version + 1

    statement = update(Dataset).where(Dataset.id == dataset_id).values(**update_data)
    await db.exec(statement)
    await db.commit()
    await db.refresh(dataset)

    logger.debug("Dataset updated", datasetId=dataset_id)
    return dataset
