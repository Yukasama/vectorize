"""AIModel service."""

import shutil
from uuid import UUID

from fastapi import Request
from loguru import logger
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.config.config import settings
from vectorize.utils.etag_parser import parse_etag

from .exceptions import ModelNotFoundError
from .models import AIModel, AIModelPublic, AIModelUpdate
from .repository import delete_model_db, get_ai_model_db, update_ai_model_db

__all__ = ["delete_model_svc", "get_ai_model_svc", "update_ai_model_svc"]


def _cleanup_model_files(model_tag: str) -> None:
    """Remove physical model files from filesystem.

    Args:
        model_tag: The model tag to determine the correct filesystem path.
    """
    try:
        # Convert database model_tag to filesystem path using same logic as
        # training/evaluation
        if model_tag.startswith("trained_models/"):
            filesystem_model_tag = model_tag
        else:
            filesystem_model_tag = model_tag.replace("_", "--")
            if not filesystem_model_tag.startswith("models--"):
                filesystem_model_tag = f"models--{filesystem_model_tag}"

        model_path = settings.model_upload_dir / filesystem_model_tag

        if model_path.exists():
            shutil.rmtree(model_path)
            logger.debug(
                "Model files deleted from filesystem", path=str(model_path)
            )
        else:
            logger.debug(
                "Model path does not exist, skipping cleanup",
                path=str(model_path),
            )

    except Exception as e:
        logger.warning(
            "Failed to cleanup model files", model_tag=model_tag, error=str(e)
        )


async def get_ai_model_svc(
    db: AsyncSession, ai_model_tag: str
) -> tuple[AIModelPublic, int]:
    """Read a single AI model from the database.

    This function retrieves an AI model by its ID from the database and returns it
    as a dictionary. The dictionary contains all fields of the AI model.

    Args:
        db: Database session for persistence operations
        ai_model_tag: The tag of the AI model to retrieve

    Returns:
        Dictionary representing the AI model with all fields.
    """
    ai_model = await get_ai_model_db(db, ai_model_tag)
    return AIModelPublic.model_validate(ai_model), ai_model.version


async def update_ai_model_svc(
    db: AsyncSession, request: Request, ai_model_id: UUID, update_data: AIModelUpdate
) -> int:
    """Update an AI model in the database.

    This function updates an existing AI model in the database with the provided
    data. It returns the updated version.

    Args:
        db: Database session for persistence operations
        request: The HTTP request object
        ai_model_id: The UUID of the AI model to update
        update_data: The updated AI model data

    Returns:
        The updated version of the AI model.
    """
    expected_version = parse_etag(str(ai_model_id), request)

    updated_model = await update_ai_model_db(
        db, ai_model_id, update_data, expected_version
    )

    logger.debug("AIModel updated", ai_model_id=ai_model_id)
    return updated_model.version


async def delete_model_svc(db: AsyncSession, model_id: UUID) -> None:
    """Delete an AI model by its ID from the database and filesystem.

    Args:
        db: Database session.
        model_id: UUID of the model to delete.
    """
    # Get model info before deletion for filesystem cleanup
    statement = select(AIModel).where(AIModel.id == model_id)
    result = await db.exec(statement)
    model = result.first()

    if model is None:
        raise ModelNotFoundError(str(model_id))

    # Delete from database first
    await delete_model_db(db, model_id)

    # Delete physical files
    _cleanup_model_files(model.model_tag)

    logger.debug(
        "Model deleted from service and filesystem", model_id=model_id
    )
