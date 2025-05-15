from uuid import UUID
from sqlmodel.ext.asyncio.session import AsyncSession

from .repository import delete_model_db
from loguru import logger

async def delete_model_srv(db: AsyncSession, model_id: UUID) -> None:
    """Delete an AI model by its ID from the database.

    Args:
        db: Database session.
        model_id: UUID of the model to delete.
    """
    await delete_model_db(db, model_id)
    logger.debug("Model deleted from service", modelId=model_id)
