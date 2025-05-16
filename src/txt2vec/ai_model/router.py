"""Test für das Hochladen eines Huggingface-Modells."""
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status
from fastapi.responses import Response
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from txt2vec.config.db import get_session

from .service import delete_model_srv

router = APIRouter()


@router.delete("/{model_id}")
async def delete_model(
    model_id: UUID, db: Annotated[AsyncSession, Depends(get_session)]
) -> Response:
    """Delete an AI model by its ID.

    Args:
        model_id: The UUID of the model to delete.
        db: Database session.

    Returns:
        204 No Content if deletion is successful.

    Raises:
        ModelNotFoundError: If no model with that ID exists.
    """
    await delete_model_srv(db, model_id)
    logger.debug("Model deleted", modelId=model_id)

    return Response(status_code=status.HTTP_204_NO_CONTENT)
