"""Actions service."""

from sqlmodel.ext.asyncio.session import AsyncSession

from .actions_model import ActionsModel
from .repository import get_actions_db
from .schemas import ActionsFilters

__all__ = ["get_actions_svc"]


async def get_actions_svc(
    db: AsyncSession, params: ActionsFilters
) -> list[ActionsModel]:
    """Retrieve and validate task actions from the database.

    This service function acts as an intermediary between the API router and the
    database repository, handling data validation and transformation. It queries
    the database for task actions based on the provided filters and converts the
    raw database rows into validated Pydantic models.

    Args:
        db: SQLModel async database session for executing queries.
        params: ActionsFilterParams object containing filter criteria such as
            limit, offset, completed status, task statuses, and time range.

    Returns:
        List of validated Pydantic models representing task actions.
        Each model contains task metadata including ID, status,
        timestamps, and task type.
    """
    rows = await get_actions_db(db, params)
    return [ActionsModel.model_validate(r, from_attributes=True) for r in rows]
