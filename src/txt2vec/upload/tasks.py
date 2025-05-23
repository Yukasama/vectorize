"""Tasks module for handling background processes.

This module contains functions for managing background tasks related to
model uploads and processing, such as handling Hugging Face models.
"""

from uuid import UUID

from loguru import logger
from sqlalchemy.exc import IntegrityError
from sqlmodel.ext.asyncio.session import AsyncSession

from txt2vec.ai_model import AIModel
from txt2vec.ai_model.models import ModelSource
from txt2vec.ai_model.repository import save_ai_model
from txt2vec.common.task_status import TaskStatus
from txt2vec.upload.exceptions import ModelAlreadyExistsError
from txt2vec.upload.github_service import load_github_model_and_cache_only
from txt2vec.upload.huggingface_service import load_model_and_cache_only
from txt2vec.upload.repository import update_upload_task_status


async def process_huggingface_model_background(
    db: AsyncSession, model_id: str, tag: str, task_id: UUID
) -> None:
    """Processes a Hugging Face model upload in the background.

    This function handles the background processing of a Hugging Face model
    upload task. It loads the model, saves it to the database, and updates
    the task status.

    Args:
        db (AsyncSession): The database session for database operations.
        model_id (str): The ID of the Hugging Face model repository.
        tag (str): The specific revision or tag of the model to download.
        task_id (UUID): The unique identifier of the upload task.

    Raises:
        Exception: If an error occurs during model processing or database
        operations.
    """
    key = f"{model_id}@{tag}"

    try:
        logger.info("[BG] Starting model upload for task", taskId=task_id)
        await load_model_and_cache_only(model_id, tag)

        ai_model = AIModel(
            model_tag=key,
            name=model_id,
            source=ModelSource.HUGGINGFACE,
        )
        await save_ai_model(db, ai_model)
        await update_upload_task_status(db, task_id, TaskStatus.DONE)

        logger.info("[BG] Task completed successfully", taskId=task_id)

    except ModelAlreadyExistsError as e:
        logger.error(f"[BG] Model already exists for task {task_id}: {e}")
        await db.rollback()
        await update_upload_task_status(
            db, task_id, TaskStatus.FAILED, error_msg=str(e)
        )
    except IntegrityError as e:
        logger.error(f"[BG] IntegrityError in task {task_id}: {e}")
        await db.rollback()
        await update_upload_task_status(
            db, task_id, TaskStatus.FAILED, error_msg=str(e)
        )
    except Exception as e:
        logger.error(f"[BG] Error in task {task_id}: {e}")
        await db.rollback()
        await update_upload_task_status(
            db, task_id, TaskStatus.FAILED, error_msg=str(e)
        )


async def process_github_model_background(
    db: AsyncSession, owner: str, repo: str, branch: str, repo_url: str, task_id: UUID
) -> None:
    """Processes a GitHub model upload in the background.

    This function handles the background processing of a GitHub model
    upload task. It loads the model, saves it to the database, and updates
    the task status.

    Args:
        db (AsyncSession): The database session for database operations.
        owner (str):
        repo (str):
        branch (str):
        repo_url (str):
        task_id (UUID): The unique identifier of the upload task.

    Raises:
        Exception: If an error occurs during model processing or database
        operations.
    """
    key = f"{owner}/{repo}@{branch}"

    try:
        logger.info("[BG] Starting model upload for task", taskId=task_id)
        await load_github_model_and_cache_only(repo_url)

        ai_model = AIModel(
            model_tag=key,
            name=repo,
            source=ModelSource.GITHUB,
        )
        await save_ai_model(db, ai_model)
        await update_upload_task_status(db, task_id, TaskStatus.DONE)

        logger.info("[BG] Task completed successfully", taskId=task_id)

    except ModelAlreadyExistsError as e:
        logger.error(f"[BG] Model already exists for task {task_id}: {e}")
        await db.rollback()
        await update_upload_task_status(
            db, task_id, TaskStatus.FAILED, error_msg=str(e)
        )
    except IntegrityError as e:
        logger.error(f"[BG] IntegrityError in task {task_id}: {e}")
        await db.rollback()
        await update_upload_task_status(
            db, task_id, TaskStatus.FAILED, error_msg=str(e)
        )
    except Exception as e:
        logger.error(f"[BG] Error in task {task_id}: {e}")
        await db.rollback()
        await update_upload_task_status(
            db, task_id, TaskStatus.FAILED, error_msg=str(e)
        )
