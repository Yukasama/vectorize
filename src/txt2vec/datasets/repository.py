"""Dataset repository."""

from uuid import UUID

from loguru import logger
from sqlmodel import select, update
from sqlmodel.ext.asyncio.session import AsyncSession

from txt2vec.datasets.exceptions import DatasetNotFoundError
from txt2vec.datasets.models import Dataset


async def save_dataset(db: AsyncSession, dataset: Dataset) -> UUID:
    """Save a new dataset to the database.

    :param db: Database session
    :param dataset: The dataset object to save
    :return: The UUID of the saved dataset
    """
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)
    logger.debug("Dataset saved to DB", dataset=dataset)
    return dataset.id


async def get_dataset(db: AsyncSession, dataset_id: UUID) -> Dataset:
    """Get a dataset by its ID.

    :param db: Database session
    :param dataset_id: The UUID of the dataset to retrieve
    :return: The dataset object
    :raises DatasetNotFoundError: If the dataset is not found
    """
    statement = select(Dataset).where(Dataset.id == dataset_id)
    result = await db.exec(statement)
    dataset = result.first()

    if dataset is None:
        raise DatasetNotFoundError(str(dataset_id))

    return dataset


async def update_dataset(
    db: AsyncSession, dataset_id: UUID, update_data: dict
) -> Dataset:
    """Update an existing dataset.

    :param db: Database session
    :param dataset_id: The UUID of the dataset to update
    :param update_data: Dictionary containing the fields to update
    :return: The updated dataset object
    :raises DatasetNotFoundError: If the dataset is not found
    """
    statement = select(Dataset).where(Dataset.id == dataset_id)
    result = await db.exec(statement)
    dataset = result.first()

    if dataset is None:
        raise DatasetNotFoundError(str(dataset_id))

    statement = update(Dataset).where(Dataset.id == dataset_id).values(**update_data)
    await db.exec(statement)
    await db.commit()

    statement = select(Dataset).where(Dataset.id == dataset_id)
    result = await db.exec(statement)
    updated_dataset = result.first()

    logger.info("Dataset updated", datasetId=dataset_id)
    return updated_dataset
