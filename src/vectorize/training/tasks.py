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
    db: AsyncSession, model_path: str, train_request: TrainRequest, task_id: UUID, dataset_paths: list[str]
) -> None:
    """Background task: trains the model and updates TrainingTask status."""
    logger.info(
        "Training started for model_path={}, dataset_paths={}, task_id={}",
        model_path,
        dataset_paths,
        task_id,
    )
    try:
        train_model_service_svc(model_path, train_request, dataset_paths)
        logger.info(
            "Training finished successfully for model_path={}, task_id={}",
            model_path,
            task_id,
        )
        asyncio_run(update_training_task_status(db, task_id, TaskStatus.DONE))
    except Exception as exc:
        logger.exception("Training failed: task_id={}", task_id)
        asyncio_run(
            update_training_task_status(db, task_id, TaskStatus.FAILED, error_msg=str(exc))
        )
