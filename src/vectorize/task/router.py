"""Tasks router."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.common.task_status import TaskStatus
from vectorize.config.db import get_session
from vectorize.task.schemas import TaskFilters

from .service import get_tasks_svc
from .tasks_model import TasksModel

__all__ = ["router"]


router = APIRouter(tags=["Tasks"])


@router.get("", summary="Get filterable tasks")
async def get_tasks(  # noqa: PLR0913, PLR0917
    db: Annotated[AsyncSession, Depends(get_session)],
    limit: Annotated[int | None, Query(ge=1, le=100)] = None,
    offset: Annotated[int | None, Query(ge=0)] = None,
    completed: Annotated[bool | None, Query()] = None,
    status: Annotated[list[TaskStatus] | None, Query()] = None,
    within_hours: Annotated[int, Query(ge=1)] = 1,
) -> list[TasksModel]:
    """Get tasks with filtering and pagination.

    Args:
        db: Database session for queries.
        limit: Maximum number of records to return (default 100).
        offset: Number of records to skip (default 0).
        completed: Filter by completion status (True/False).
        status: Filter by specific task statuses.
        within_hours: Time window in hours to filter tasks (default 1).

    Returns:
        List of task action models with metadata.
    """
    task_filters = TaskFilters(
        limit=limit,
        offset=offset,
        completed=completed,
        statuses=status,
        within_hours=within_hours,
    )
    logger.debug("Fetching tasks with parameters", filters=str(task_filters))
    return await get_tasks_svc(db, task_filters)
