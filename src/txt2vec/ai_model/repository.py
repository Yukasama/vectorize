"""Embeddings repository."""

from uuid import UUID

from loguru import logger
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from .exceptions import ModelNotFoundError
from .models import AIModel
from .utils.tag_helpers import next_available_tag


async def get_ai_model(db: AsyncSession, model_id: UUID) -> AIModel:
    """Retrieve an AI model by its ID.

    Args:
        db: Database session instance.
        model_id: The UUID of the model to retrieve.

    Returns:
        AIModel: The AI model object corresponding to the given ID.

    Raises:
        ModelNotFoundError: If the model is not found.
    """
    statement = select(AIModel).where(AIModel.id == model_id)
    result = await db.exec(statement)
    model = result.first()

    if model is None:
        raise ModelNotFoundError(str(model_id))

    return model


async def save_ai_model(db: AsyncSession, model: AIModel) -> UUID:
    """Persist model to database with a unique tag.

    Args:
        db: AsyncSession
            Database session instance.
        model: AIModel
            The AI model object to save.

    Returns:
        UUID: The ID of the saved AI model.
    """
    model.model_tag = await next_available_tag(db, model.model_tag)

    db.add(model)
    await db.commit()
    await db.refresh(model)

    logger.debug("AI model saved to DB: {}", model.id)
    return model.id
