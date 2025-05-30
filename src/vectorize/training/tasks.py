"""Background task for model training."""

from asyncio import run as asyncio_run
from uuid import UUID

from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.common.task_status import TaskStatus

from .repository import update_training_task_status
from .schemas import TrainRequest
from .service import train_model_service_svc


def train_model_task(
    db: AsyncSession, model_id: str, train_request: TrainRequest, task_id: UUID
) -> None:
    """Background task: trains the model and updates TrainingTask status."""
    logger.info(
        "Training started for model_id=%s, dataset_paths=%s, task_id=%s",
        model_id,
        train_request.dataset_paths,
        task_id,
    )
    try:
        train_model_service_svc(model_id, train_request)
        logger.info(
            "Training finished successfully for model_id=%s, task_id=%s",
            model_id,
            task_id,
        )
        asyncio_run(update_training_task_status(db, task_id, TaskStatus.DONE))
    except Exception as exc:
        logger.error(
            "Training failed: %s, task_id=%s",
            str(exc),
            task_id,
        )
        asyncio_run(
            update_training_task_status(
                db, task_id, TaskStatus.FAILED, error_msg=str(exc)
            )
        )
