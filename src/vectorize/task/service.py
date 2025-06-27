"""Taks service."""

from sqlmodel.ext.asyncio.session import AsyncSession

from .repository import get_tasks_db
from .schemas import TaskFilters
from .tasks_model import TaskModel

__all__ = ["get_tasks_svc"]


async def get_tasks_svc(db: AsyncSession, params: TaskFilters) -> list[TaskModel]:
    """Retrieve and validate tasks from the database.

    This service function acts as an intermediary between the API router and the
    database repository, handling data validation and transformation. It queries
    the database for tasks based on the provided filters and converts the
    raw database rows into validated Pydantic models.

    Args:
        db: SQLModel async database session for executing queries.
        params: TaskFilters object containing filter criteria such as
            limit, offset, completed status, task statuses, and time range.

    Returns:
        List of validated Pydantic models representing tasks.
        Each model contains task metadata including ID, status,
        timestamps, and task type.
    """
    rows = await get_tasks_db(db, params)
    return [TaskModel.model_validate(r, from_attributes=True) for r in rows]
