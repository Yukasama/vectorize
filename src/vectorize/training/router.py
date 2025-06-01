"""Training router."""


from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, Response, status
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.common.task_status import TaskStatus
from vectorize.config.db import get_session

from .exceptions import TrainingDatasetNotFoundError, TrainingTaskNotFoundError
from .models import TrainingTask
from .repository import get_train_task_by_id, save_training_task
from .schemas import TrainRequest, TrainingStatusResponse
from .tasks import train_model_task
from .utils.helpers import get_dataset_paths_by_ids, get_model_path_by_id

__all__ = ["router"]

router = APIRouter(tags=["Training"])


@router.post("/train")
async def train_model(
    train_request: TrainRequest,
    background_tasks: BackgroundTasks,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    """Start model training as a background task and persist TrainingTask."""
    dataset_paths = await get_dataset_paths_by_ids(db, train_request.dataset_ids)
    missing = [p for p in dataset_paths if not Path(p).is_file()]
    if missing:
        for p in missing:
            logger.exception("Training request failed: Dataset file not found: {}", p)
        raise TrainingDatasetNotFoundError(missing[0])
    model_path = await get_model_path_by_id(db, train_request.model_id)
    logger.info(
        "Training requested for model_id={}, model_path={}, dataset_paths={}",
        train_request.model_id,
        model_path,
        dataset_paths,
    )
    task = TrainingTask(id=uuid4(), task_status=TaskStatus.PENDING)
    await save_training_task(db, task)
    background_tasks.add_task(
        train_model_task,
        db,
        model_path,
        train_request,
        task.id,
        dataset_paths,
    )
    logger.info(
        "Training started in background for model_id={}, task_id={}",
        train_request.model_id,
        str(task.id),
    )
    return Response(status_code=status.HTTP_202_ACCEPTED)


@router.get("/{task_id}/status")
async def get_training_status(
    task_id: UUID,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> TrainingStatusResponse:
    """Get the status and metadata of a training task by its ID."""
    task = await get_train_task_by_id(db, task_id)
    if not task:
        raise TrainingTaskNotFoundError(str(task_id))
    return TrainingStatusResponse.from_task(task)
