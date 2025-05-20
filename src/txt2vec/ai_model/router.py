"""AIModel router."""

from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    Request,
    Response,
    status,
)
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from txt2vec.config.db import get_session

from .models import AIModelUpdate
from .service import update_ai_model_srv

__all__ = ["router"]


router = APIRouter(tags=["AIModel"])


@router.put("/{ai_model_id}")
async def update_ai_model(
    ai_model_id: UUID,
    request: Request,
    ai_model: AIModelUpdate,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    """Update an AI model with version control using ETags.

    Updates an AI model by its ID, requiring an If-Match header with the current
    version to prevent concurrent modification issues.

    Args:
        ai_model_id: The UUID of the AIModel to update
        request: The HTTP request object containing If-Match header
        response: FastAPI response object for setting headers
        ai_model: The updated AIModel object
        db: Database session for persistence operations

    Returns:
        204 No Content response with Location header

    Raises:
        VersionMismatchError: If the ETag doesn't match current version
        VersionMissingError: If the If-Match header is missing
        ModelNotFoundError: If the AIModel doesn't exist
    """
    new_version = await update_ai_model_srv(db, request, ai_model_id, ai_model)
    logger.debug("AIModel updated", ai_model_id=ai_model_id)

    return Response(
        status_code=status.HTTP_204_NO_CONTENT,
        headers={"Location": f"{request.url.path}", "ETag": f'"{new_version}"'},
    )
