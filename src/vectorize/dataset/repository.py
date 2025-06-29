"""Dataset repository."""

from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID

from loguru import logger
from sqlmodel import func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.common.exceptions import VersionMismatchError
from vectorize.task.exceptions import TaskNotFoundError
from vectorize.task.task_status import TaskStatus

from .dataset_source import DatasetSource
from .exceptions import DatasetNotFoundError
from .models import Dataset, DatasetUpdate
from .task_model import UploadDatasetTask

__all__ = [
    "delete_dataset_db",
    "find_dataset_by_name_db",
    "get_dataset_db",
    "get_datasets_db",
    "get_upload_dataset_task_db",
    "is_dataset_being_uploaded_db",
    "save_upload_dataset_task_db",
    "update_dataset_db",
    "update_upload_task_status_db",
    "upload_dataset_db",
]


async def get_datasets_db(
    db: AsyncSession, *, limit: int, offset: int
) -> tuple[Sequence[Dataset], int]:
    """Retrieve all datasets from the database.

    Args:
        db: Database session instance.
        limit: Maximum number of datasets to return.
        offset: Number of datasets to skip.

    Returns:
        A paged list of all Dataset objects in the database.
    """
    stmt = select(Dataset).offset(offset).limit(limit)
    datasets = (await db.exec(stmt)).all()

    total_stmt = select(func.count()).select_from(Dataset)
    total: int = await db.scalar(total_stmt)

    logger.debug("Datasets retrieved", items=len(datasets), total=total)
    return datasets, total


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


async def find_dataset_by_name_db(db: AsyncSession, name: str) -> Dataset | None:
    """Retrieve a dataset by its ID.

    Args:
        db: Database session instance.
        name: The name prefix of the dataset.

    Returns:
        Dataset: The dataset object corresponding to the given ID.

    Raises:
        DatasetNotFoundError: If the dataset is not found.
    """
    statement = select(Dataset).where(
        (Dataset.name.startswith(name)) & (Dataset.source == DatasetSource.HUGGINGFACE)
    )
    result = await db.exec(statement)
    return result.first()


async def is_dataset_being_uploaded_db(db: AsyncSession, dataset_tag: str) -> bool:
    """Check if a dataset upload task is currently running for the given tag.

    This function queries the database to determine if there is an active
    upload task (with RUNNING status) for the specified dataset tag.

    Args:
        db: Database session instance for executing queries.
        dataset_tag: The tag identifier of the dataset to check.

    Returns:
        bool: True if there is a running upload task for the dataset tag.
    """
    running_task_stmt = select(UploadDatasetTask).where(
        (UploadDatasetTask.tag == dataset_tag)
        & (UploadDatasetTask.task_status == TaskStatus.RUNNING)
    )
    running_task_result = await db.exec(running_task_stmt)
    return running_task_result.first() is not None


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


async def save_upload_dataset_task_db(
    db: AsyncSession, task: UploadDatasetTask
) -> None:
    """Saves a new upload dataset task to the database.

    Args:
        db: The asynchronous database session.
        task: The upload dataset task to be saved.
    """
    db.add(task)
    await db.commit()
    await db.refresh(task)
    logger.debug("Save dataset task saved to DB", task=task)


async def get_upload_dataset_task_db(
    db: AsyncSession, task_id: UUID
) -> UploadDatasetTask:
    """Retrieves an upload dataset task by its ID.

    Args:
        db: The asynchronous database session.
        task_id: The unique identifier of the upload task.

    Returns:
        UploadDatasetTask: The upload task object if found, otherwise None.
    """
    result = await db.exec(
        select(UploadDatasetTask).where(UploadDatasetTask.id == task_id)
    )
    task = result.first()

    if task is None:
        logger.debug("Upload task not found", task_id=task_id)
        raise TaskNotFoundError(task_id)

    logger.debug("Upload task retrieved", task=task)
    return task


async def update_upload_task_status_db(
    db: AsyncSession,
    task_id: UUID,
    status: TaskStatus,
    error_msg: str | None = None,
) -> None:
    """Updates the status of an existing upload task in the database.

    Args:
        db: The asynchronous database session.
        task_id: The unique identifier of the upload task.
        status: The new status to set for the task.
        error_msg: An optional error message if the task failed.
    """
    result = await db.exec(
        select(UploadDatasetTask).where(UploadDatasetTask.id == task_id)
    )
    task = result.first()
    if task is None:
        logger.error("Upload task not found", task_id=task_id)
        raise ValueError(f"Upload task {task_id} not found")

    task.task_status = status
    task.end_date = datetime.now(tz=UTC)
    task.error_msg = error_msg

    db.add(task)
    await db.commit()
    logger.debug("Update Dataset task status", task=task)


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


async def delete_dataset_db(db: AsyncSession, dataset_id: UUID) -> str | None:
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
    return dataset.file_name
