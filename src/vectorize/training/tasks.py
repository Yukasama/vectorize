"""Background task for model training."""

from uuid import UUID

from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.common.task_status import TaskStatus

from .exceptions import TrainingModelNotFoundError
from .repository import update_training_task_status
from .schemas import TrainRequest
from .service import train_model_service_svc


def train_model_task(db: AsyncSession, train_request: TrainRequest, task_id: UUID) -> None:
    """Background task: trains the model and updates TrainingTask status."""
    from asyncio import run as asyncio_run

    logger.info(
        "[BG] Training started for model_path={}, dataset_paths={}, task_id={}",
        train_request.model_path,
        train_request.dataset_paths,
        task_id,
    )
    try:
        train_model_service_svc(train_request)
        logger.info(
            "[BG] Training finished successfully for model_path={}, task_id={}",
            train_request.model_path,
            task_id,
        )
        # Update status to DONE
        asyncio_run(update_training_task_status(db, task_id, TaskStatus.DONE))
    except TrainingModelNotFoundError:
        logger.error(
            "[BG] Training failed: Invalid model path: {}, task_id={}",
            train_request.model_path,
            task_id,
        )
        asyncio_run(
            update_training_task_status(
                db, task_id, TaskStatus.FAILED, error_msg="Invalid model path"
            )
        )
    except Exception as e:
        logger.error(
            "[BG] Training failed for model_path={}, task_id={}: {}",
            train_request.model_path,
            task_id,
            e,
        )
        asyncio_run(update_training_task_status(db, task_id, TaskStatus.FAILED, error_msg=str(e)))
        raise
