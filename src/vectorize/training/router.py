"""Training router for SBERT triplet training (Hugging Face TRL)."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Response, status
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.ai_model.exceptions import ModelNotFoundError
from vectorize.ai_model.repository import get_ai_model_db
from vectorize.common.task_status import TaskStatus
from vectorize.config.config import settings
from vectorize.config.db import get_session

from .exceptions import (
    TrainingModelWeightsNotFoundError,
    TrainingTaskNotFoundError,
)
from .models import TrainingTask
from .repository import (
    get_train_task_by_id,
    save_training_task,
    update_training_task_status,
)
from .schemas import TrainRequest, TrainingStatusResponse
from .service import TrainingOrchestrator
from .tasks import run_training_bg

__all__ = ["router"]

router = APIRouter(tags=["Training"])


def has_model_weights(path: str) -> bool:
    """Check if model directory contains weight files."""
    model_path = Path(path)
    return any(model_path.glob("**/*.safetensors")) or any(model_path.glob("**/*.bin"))


@router.post("/train")
async def train_model(
    train_request: TrainRequest,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    """Starts SBERT triplet training as a background task.

    Expects JSONL with question, positive, negative.
    If no val_dataset_id is provided,
    10% of the first training dataset will be used for validation.
    """
    model = await get_ai_model_db(db, train_request.model_tag)
    if not model:
        raise ModelNotFoundError(train_request.model_tag)
    model_path = str(settings.model_upload_dir / model.model_tag)

    if not has_model_weights(model_path):
        raise TrainingModelWeightsNotFoundError(
            f"Model weights not found in {model_path} (searched recursively)"
        )

    dataset_paths = await TrainingOrchestrator.validate_datasets(
        db, train_request.train_dataset_ids, train_request.val_dataset_id
    )
    tag_time = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    task = TrainingTask(id=uuid4())
    output_dir = (
        f"data/models/trained_models/{model.model_tag}-finetuned-"
        f"{tag_time}-{str(task.id)[:8]}"
    )
    logger.debug(
        "SBERT-Triplet-Training requested."
        "model_tag={}, dataset_count={}, model_path={}",
        train_request.model_tag,
        len(dataset_paths),
        model_path,
    )
    await save_training_task(db, task)

    run_training_bg.send(
        model_path=model_path,
        train_request_dict=train_request.model_dump(),
        task_id=str(task.id),
        dataset_paths=dataset_paths,
        output_dir=output_dir,
    )
    logger.debug(
        "SBERT-Triplet-Training started in background with Dramatiq.",
        task_id=str(task.id),
        model_tag=train_request.model_tag,
        dataset_count=len(dataset_paths),
        model_path=model_path,
    )
    location = f"/training/{task.id}/status"
    return Response(
        status_code=status.HTTP_202_ACCEPTED,
        headers={"Location": location},
    )


@router.get("/{task_id}/status")
async def get_training_status(
    task_id: UUID,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> TrainingStatusResponse:
    """Returns the status and metadata of a training job."""
    task = await get_train_task_by_id(db, task_id)
    if not task:
        raise TrainingTaskNotFoundError(str(task_id))
    return TrainingStatusResponse.from_task(task)


@router.post("/{task_id}/cancel")
async def cancel_training(
    task_id: UUID,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    """Cancel a running training task."""
    task = await get_train_task_by_id(db, task_id)
    if not task:
        raise TrainingTaskNotFoundError(str(task_id))
    if task.task_status not in {TaskStatus.QUEUED, TaskStatus.RUNNING}:
        return Response(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=f"Cannot cancel task with status: {task.task_status.name}",
        )
    await update_training_task_status(
        db, task_id, TaskStatus.CANCELED, error_msg="Training canceled by user"
    )
    logger.debug(
        "Training task marked for cancellation.",
        task_id=task_id,
    )
    return Response(status_code=status.HTTP_200_OK)
