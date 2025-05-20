"""AIModel service."""

from uuid import UUID

from fastapi import Request
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from txt2vec.utils.etag_parser import parse_etag

from .models import AIModelUpdate
from .repository import update_ai_model_db

__all__ = ["update_ai_model_srv"]


async def update_ai_model_srv(
    db: AsyncSession,
    request: Request,
    model_id: UUID,
    update_data: AIModelUpdate,
) -> int:
    """Update an AI model in the database.

    This function updates an existing AI model in the database with the provided
    data. It returns the updated version.

    Args:
        db: Database session for persistence operations
        request: The HTTP request object
        model_id: The UUID of the AI model to update
        update_data: The updated AI model data

    Returns:
        The updated version of the AI model.
    """
    expected_version = parse_etag(str(model_id), request)

    updated_model = await update_ai_model_db(
        db, model_id, update_data, expected_version
    )

    logger.debug("AIModel updated", ai_model_id=model_id)
    return updated_model.version
