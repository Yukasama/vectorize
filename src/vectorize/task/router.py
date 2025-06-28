"""Tasks router."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.config.db import get_session
from vectorize.task.schemas import TaskFilters
from vectorize.task.task_status import TaskStatus
from vectorize.task.task_type import TaskType

from .service import get_tasks_svc
from .tasks_model import TaskModel

__all__ = ["router"]


router = APIRouter(tags=["Tasks"])


@router.get("", summary="Get filterable tasks")
async def get_tasks(  # noqa: PLR0913, PLR0917
    db: Annotated[AsyncSession, Depends(get_session)],
    limit: Annotated[int | None, Query(ge=1, le=100)] = None,
    offset: Annotated[int | None, Query(ge=0)] = None,
    tag: Annotated[str | None, Query(max_length=100)] = None,
    task_type: Annotated[list[TaskType] | None, Query()] = None,
    status: Annotated[list[TaskStatus] | None, Query()] = None,
    within_hours: Annotated[int, Query(ge=1)] = 1,
) -> list[TaskModel]:
    """Get tasks with filtering and pagination.

    Args:
        db: Database session for queries.
        limit: Maximum number of records to return (default 100).
        offset: Number of records to skip (default 0).
        tag: Filter tasks by specific tag.
        task_type: Filter tasks by specific type (e.g., upload, synthesis).
        status: Filter by specific task statuses.
        within_hours: Time window in hours to filter tasks (default 1).

    Returns:
        List of task action models with metadata.
    """
    task_filters = TaskFilters(
        limit=limit,
        offset=offset,
        tag=tag,
        task_types=task_type,
        statuses=status,
        within_hours=within_hours,
    )
    logger.debug("Fetching tasks with parameters", filters=str(task_filters))
    return await get_tasks_svc(db, task_filters)
