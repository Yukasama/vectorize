"""Dataset repository."""

from uuid import UUID

from loguru import logger
from sqlmodel import select, update

from txt2vec.config.db import session
from txt2vec.datasets.exceptions import DatasetNotFoundError
from txt2vec.datasets.models import Dataset


async def save_dataset(dataset: Dataset) -> UUID:
    """Save a new dataset to the database.

    :param dataset: The dataset object to save
    :return: The UUID of the saved dataset
    """
    async with session() as db_session:
        db_session.add(dataset)
        await db_session.commit()
        await db_session.refresh(dataset)
        logger.debug("Dataset saved", dataset=dataset)
        return dataset.id


async def get_dataset(dataset_id: UUID) -> Dataset:
    """Get a dataset by its ID.

    :param dataset_id: The UUID of the dataset to retrieve
    :return: The dataset object
    :raises DatasetNotFoundError: If the dataset is not found
    """
    async with session() as db_session:
        statement = select(Dataset).where(Dataset.id == dataset_id)
        result = await db_session.exec(statement)
        dataset = result.first()

        if dataset is None:
            raise DatasetNotFoundError(dataset_id)

        return dataset


async def update_dataset(dataset_id: UUID, update_data: dict) -> Dataset:
    """Update an existing dataset.

    :param dataset_id: The UUID of the dataset to update
    :param update_data: Dictionary containing the fields to update
    :return: The updated dataset object
    :raises DatasetNotFoundError: If the dataset is not found
    """
    async with session() as db_session:
        statement = select(Dataset).where(Dataset.id == dataset_id)
        result = await db_session.exec(statement)
        dataset = result.first()

        if dataset is None:
            raise DatasetNotFoundError(dataset_id)

        statement = (
            update(Dataset).where(Dataset.id == dataset_id).values(**update_data)
        )
        await db_session.exec(statement)
        await db_session.commit()

        statement = select(Dataset).where(Dataset.id == dataset_id)
        result = await db_session.exec(statement)
        updated_dataset = result.first()

        logger.info("Dataset updated", datasetId=dataset.id)
        return updated_dataset


async def delete_dataset(dataset_id: UUID) -> bool:
    """Delete a dataset by its ID.

    :param dataset_id: The UUID of the dataset to delete
    :return: True if deletion was successful
    :raises DatasetNotFoundError: If the dataset is not found
    """
    async with session() as db_session:
        statement = select(Dataset).where(Dataset.id == dataset_id)
        result = await db_session.exec(statement)
        dataset = result.first()

        if dataset is None:
            raise DatasetNotFoundError(dataset_id)

        await db_session.delete(dataset)
        await db_session.commit()

        logger.info("Dataset deleted", datasetId=dataset.id)
        return True


async def list_datasets(limit: int = 100, skip: int = 0) -> list[Dataset]:
    """List all datasets with pagination.

    :param limit: Maximum number of datasets to return
    :param skip: Number of datasets to skip
    :return: List of dataset objects
    """
    async with session() as db_session:
        statement = select(Dataset).offset(skip).limit(limit)
        result = await db_session.exec(statement)
        datasets = result.all()

        logger.info("Dataset deleted", datasets=len(datasets))
        return datasets


async def get_dataset_by_name(name: str) -> Dataset:
    """Get a dataset by its name.

    :param name: The name of the dataset to retrieve
    :return: The dataset object
    :raises DatasetNotFoundError: If the dataset is not found
    """
    async with session() as db_session:
        statement = select(Dataset).where(Dataset.name == name)
        result = await db_session.exec(statement)
        dataset = result.first()

        if dataset is None:
            raise DatasetNotFoundError(name)

        return dataset
