"""Repository for synthesis task operations."""

from datetime import UTC, datetime
from uuid import UUID

from loguru import logger
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.common.task_status import TaskStatus

from .models import SynthesisTask

__all__ = [
    "get_synthesis_task_by_id",
    "get_synthesis_tasks",
    "save_synthesis_task",
    "update_synthesis_task_status",
]


async def save_synthesis_task(db: AsyncSession, task: SynthesisTask) -> SynthesisTask:
    """Save a synthesis task to the database.

    Args:
        db: Database session
        task: The synthesis task to save

    Returns:
        The saved synthesis task with ID
    """
    db.add(task)
    await db.commit()
    await db.refresh(task)
    logger.debug("Synthesis task saved to DB", task_id=task.id)
    return task


async def get_synthesis_task_by_id(
    db: AsyncSession, task_id: UUID
) -> SynthesisTask | None:
    """Get a synthesis task by ID.

    Args:
        db: Database session
        task_id: ID of the synthesis task

    Returns:
        The synthesis task or None if not found
    """
    statement = select(SynthesisTask).where(SynthesisTask.id == task_id)
    result = await db.exec(statement)
    task = result.first()

    if task:
        logger.debug("Synthesis task loaded from DB", task_id=task_id)

    return task


async def get_synthesis_tasks(db: AsyncSession, limit: int = 20) -> list[SynthesisTask]:
    """Get a list of synthesis tasks.

    Args:
        db: Database session
        limit: Maximum number of tasks to return

    Returns:
        List of synthesis tasks ordered by creation date (newest first)
    """
    statement = (
        select(SynthesisTask).order_by(SynthesisTask.created_at.desc()).limit(limit)
    )
    result = await db.exec(statement)
    tasks = result.all()

    logger.debug("Retrieved synthesis tasks from DB", count=len(tasks))
    return tasks


async def update_synthesis_task_status(
    db: AsyncSession,
    task_id: UUID,
    status: TaskStatus,
    error_msg: str | None = None,
) -> None:
    """Update the status of a synthesis task.

    Args:
        db: Database session
        task_id: ID of the synthesis task
        status: New status for the task
        error_msg: Optional error message (for failed tasks)
    """
    task = await get_synthesis_task_by_id(db, task_id)
    if not task:
        logger.error("Cannot update status - Synthesis task not found", task_id=task_id)
        return

    task.task_status = status

    if error_msg:
        task.error_msg = error_msg

    # Bei erfolgreichem Abschluss oder Fehlschlag setzen wir auch das Enddatum
    if status in {TaskStatus.DONE, TaskStatus.FAILED}:
        task.end_date = datetime.now(tz=UTC).date()

    await db.commit()
    await db.refresh(task)

    logger.debug(
        "Synthesis task status updated",
        task_id=task_id,
        status=status,
        error_msg=error_msg,
    )
