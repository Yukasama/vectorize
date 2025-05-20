"""Repository module for managing upload tasks in the database.

This module provides functions to save and update upload tasks in the database
using SQLModel and asynchronous database sessions.
"""

from datetime import UTC, datetime
from uuid import UUID

from loguru import logger
from sqlalchemy.exc import NoResultFound
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from txt2vec.common.status import TaskStatus
from txt2vec.upload.models import UploadTask


async def save_upload_task(db: AsyncSession, task: UploadTask) -> None:
    """Saves a new upload task to the database.

    Args:
        db (AsyncSession): The asynchronous database session.
        task (UploadTask): The upload task to be saved.
    """
    db.add(task)
    await db.commit()
    await db.refresh(task)


async def update_upload_task_status(
    db: AsyncSession,
    task_id: UUID,
    status: TaskStatus,
    error_msg: str | None = None,
) -> None:
    """Updates the status of an existing upload task in the database.

    Args:
        db (AsyncSession): The asynchronous database session.
        task_id (UUID): The unique identifier of the upload task.
        status (TaskStatus): The new status to set for the task.
        error_msg (str | None): An optional error message if the task failed.

    Hinweis: Falls der Task nicht gefunden wird, wird ein Fehler geloggt,
    aber keine Exception geworfen.
    """
    result = await db.exec(select(UploadTask).where(UploadTask.id == task_id))
    try:
        task = result.one()
        task.task_status = status
        task.end_date = datetime.now(tz=UTC)
        task.error_msg = error_msg
        await db.commit()
    except NoResultFound:
        logger.exception(
            "[update_upload_task_status] Task mit ID nicht gefunden!", task_id=task_id
        )


async def create_upload_task(db: AsyncSession, model_tag: str, source: str) -> UploadTask:
    task = UploadTask(
        model_tag=model_tag,
        source=source,
        task_status=TaskStatus.PENDING,
        start_date=datetime.now(tz=UTC),
    )
    db.add(task)
    await db.commit()
    await db.refresh(task)
    return task
