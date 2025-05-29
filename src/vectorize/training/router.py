"""Training router."""


from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, Response, status
from fastapi.responses import JSONResponse
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.common.task_status import TaskStatus
from vectorize.config.db import get_session

from .exceptions import TrainingDatasetNotFoundError
from .models import TrainingTask
from .repository import save_training_task
from .schemas import TrainRequest
from .tasks import train_model_task

__all__ = ["router"]

router = APIRouter(tags=["Training"])


@router.post("/train")
async def train_model(
    train_request: TrainRequest,
    background_tasks: BackgroundTasks,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    """Start model training as a background task and persist TrainingTask."""
    missing = [p for p in train_request.dataset_paths if not Path(p).is_file()]
    if missing:
        for p in missing:
            logger.error("Training request failed: Dataset file not found: %s", p)
        raise TrainingDatasetNotFoundError(missing[0])
    logger.info(
        "Training requested for model_path=%s, dataset_paths=%s",
        train_request.model_path,
        train_request.dataset_paths,
    )
    task = TrainingTask(id=uuid4(), task_status=TaskStatus.PENDING)
    await save_training_task(db, task)
    background_tasks.add_task(train_model_task, db, train_request, task.id)
    logger.info(
        "Training started in background for model_path=%s, task_id=%s",
        train_request.model_path,
        str(task.id),
    )
    return JSONResponse(
        content={
            "message": "Training started",
            "model_path": train_request.model_path,
            "task_id": str(task.id),
        },
        status_code=status.HTTP_202_ACCEPTED,
    )
