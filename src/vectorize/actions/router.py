"""Actions router."""

from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
)
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.actions.service import get_actions_svc
from vectorize.config.db import get_session

router = APIRouter(tags=["Actions"])


@router.get("", summary="Get all current actions")
async def get_actions(
    db: Annotated[AsyncSession, Depends(get_session)],
) -> list:
    """Get all current actions.

    Returns a list of all current actions.
    """
    return await get_actions_svc(db)
