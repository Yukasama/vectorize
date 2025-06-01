"""Repository functions for TrainingTask persistence."""

from datetime import UTC, datetime
from uuid import UUID

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.common.task_status import TaskStatus

from .models import TrainingTask

__all__ = [
    "get_train_task_by_id",
    "save_training_task",
    "update_training_task_status",
    "update_training_task_progress",
]


async def save_training_task(db: AsyncSession, task: TrainingTask) -> None:
    """Persist a new TrainingTask to the database."""
    db.add(task)
    await db.commit()
    await db.refresh(task)


async def update_training_task_status(
    db: AsyncSession, task_id: UUID, status: TaskStatus, error_msg: str | None = None
) -> None:
    """Update the status and error message of a TrainingTask."""
    result = await db.exec(select(TrainingTask).where(TrainingTask.id == task_id))
    task = result.first()
    if task:
        task.task_status = status
        if status in {TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELED}:
            task.end_date = datetime.now(UTC)
        if error_msg:
            task.error_msg = error_msg
        await db.commit()
        await db.refresh(task)


async def update_training_task_progress(
    db: AsyncSession, task_id: UUID, progress: float
) -> None:
    """Update the progress of a TrainingTask."""
    result = await db.exec(select(TrainingTask).where(TrainingTask.id == task_id))
    task = result.first()
    if task:
        task.progress = progress
        await db.commit()
        await db.refresh(task)


async def get_train_task_by_id(db: AsyncSession, task_id: UUID) -> TrainingTask | None:
    """Fetch a TrainingTask by its ID."""
    result = await db.exec(select(TrainingTask).where(TrainingTask.id == task_id))
    return result.first()
