"""Training router for DPO training (Hugging Face TRL)."""

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, Response, status
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.ai_model.exceptions import ModelNotFoundError
from vectorize.ai_model.repository import get_ai_model_by_id
from vectorize.common.task_status import TaskStatus
from vectorize.config.db import get_session
from vectorize.datasets.repository import get_dataset_db

from .exceptions import (
    InvalidDatasetIdError,
    InvalidModelIdError,
    TrainingDatasetNotFoundError,
    TrainingTaskNotFoundError,
)
from .models import TrainingTask
from .repository import get_train_task_by_id, save_training_task
from .schemas import TrainRequest, TrainingStatusResponse
from .tasks import train_model_task
from .utils.uuid_utils import is_valid_uuid

__all__ = ["router"]

router = APIRouter(tags=["Training"])


@router.post("/train")
async def train_model(
    train_request: TrainRequest,
    background_tasks: BackgroundTasks,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    """Starts DPO training as a background task. Expects prompt/chosen/rejected data."""
    if not is_valid_uuid(train_request.model_id):
        raise InvalidModelIdError(train_request.model_id)
    model = await get_ai_model_by_id(db, UUID(train_request.model_id))
    if not model:
        raise ModelNotFoundError(train_request.model_id)
    model_path = str(Path("data/models") / model.model_tag)
    model_weights_path = Path(model_path)
    if not any(model_weights_path.glob("*.bin")) and not any(
        model_weights_path.glob("*.safetensors")
    ):
        raise TrainingDatasetNotFoundError(
            f"Model weights not found in {model_path}"
        )
    dataset_paths = []
    for ds_id in train_request.dataset_ids:
        if not is_valid_uuid(ds_id):
            raise InvalidDatasetIdError(ds_id)
        ds_uuid = uuid.UUID(ds_id)
        ds = await get_dataset_db(db, ds_uuid)
        dataset_path = Path("data/datasets") / ds.file_name
        dataset_paths.append(str(dataset_path))
    missing = [str(p) for p in dataset_paths if not Path(p).is_file()]
    if missing:
        raise TrainingDatasetNotFoundError(
            f"Missing datasets: {', '.join(missing)}"
        )
    # Set output_dir automatically based on model and timestamp
    tag_time = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    output_dir = (
        f"data/models/trained_models/{model.model_tag}-finetuned-{tag_time}"
    )

    logger.bind(
        model_id=train_request.model_id,
        dataset_count=len(dataset_paths),
        model_path=model_path
    ).info(
        "DPO-Training requested.")
    task = TrainingTask(id=uuid4(), task_status=TaskStatus.PENDING)
    await save_training_task(db, task)
    background_tasks.add_task(
        train_model_task,
        db,
        model_path,
        train_request,
        task.id,
        [str(p) for p in dataset_paths],
        output_dir,  # pass output_dir to task
    )
    logger.bind(
        task_id=str(task.id),
        model_id=train_request.model_id,
        dataset_count=len(dataset_paths),
        model_path=model_path
    ).info(
        "DPO-Training started in background.")
    return Response(status_code=status.HTTP_202_ACCEPTED)


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
