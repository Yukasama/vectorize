"""AI-Model repository."""

from uuid import UUID

from loguru import logger
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from .exceptions import ModelNotFoundError
from .models import AIModel

__all__ = ["get_ai_model", "save_ai_model"]


async def get_ai_model(db: AsyncSession, model_tag: str) -> AIModel:
    """Retrieve an AI model by its ID.

    Args:
        db: Database session instance.
        model_tag: The Tag of the model to retrieve.

    Returns:
        AIModel: The AI model object corresponding to the given Model Tag.

    Raises:
        ModelNotFoundError: If the model is not found.
    """
    statement = select(AIModel).where(AIModel.model_tag == model_tag)
    result = await db.exec(statement)
    model = result.first()

    if model is None:
        raise ModelNotFoundError(str(model_tag))

    logger.debug("AI Model loaded from DB", model=model)

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
    db.add(model)
    await db.commit()
    await db.refresh(model)

    logger.debug("AI Model saved to DB: {}", model.id)
    return model.id
