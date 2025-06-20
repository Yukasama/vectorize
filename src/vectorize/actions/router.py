"""Actions router."""

from typing import Annotated

from fastapi import APIRouter, Depends
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.actions.schemas import ActionQueryParams, ActionsFilterParams
from vectorize.config.db import get_session

from .actions_model import ActionsModel
from .service import get_actions_svc

__all__ = ["router"]


router = APIRouter(tags=["Actions"])


@router.get("", summary="Get filterable task actions")
async def get_actions(
    db: Annotated[AsyncSession, Depends(get_session)],
    params: Annotated[ActionQueryParams, Depends()],
) -> list[ActionsModel]:
    """Get task actions with filtering and pagination.

    Args:
        db: Database session for queries.
        params: ActionsParams object containing filter criteria such as limit,
            offset, completed status, task statuses, and time range.

    Returns:
        List of task action models with metadata.
    """
    logger.debug("Received request to get actions", params=str(params))
    actions_params = ActionsFilterParams(
        limit=params.limit,
        offset=params.offset,
        completed=params.completed,
        statuses=params.status,
        within_hours=params.within_hours,
    )
    logger.debug("Fetching actions with parameters", params=str(actions_params))
    return await get_actions_svc(db, actions_params)
