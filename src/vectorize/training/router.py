"""Training router für DPO-Training (Hugging Face TRL)."""

import uuid
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, Response, status
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.ai_model.repository import get_ai_model_by_id
from vectorize.common.task_status import TaskStatus
from vectorize.config.db import get_session
from vectorize.datasets.repository import get_dataset_db

from .exceptions import (
    InvalidModelIdError,
    InvalidDatasetIdError,
    TrainingDatasetNotFoundError,
    TrainingTaskNotFoundError,
)
from .models import TrainingTask
from .repository import get_train_task_by_id, save_training_task
from .schemas import TrainRequest, TrainingStatusResponse
from .tasks import train_model_task
from .utils.uuid_utils import is_valid_uuid, normalize_uuid

__all__ = ["router"]

router = APIRouter(tags=["Training"])


@router.post("/train")
async def train_model(
    train_request: TrainRequest,
    background_tasks: BackgroundTasks,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    """Startet DPO-Training als Background-Task. Erwartet Datensätze im prompt/chosen/rejected-Format."""
    # Validierung der Modell-ID
    if not is_valid_uuid(train_request.model_id):
        raise InvalidModelIdError(train_request.model_id)
    norm_model_id = normalize_uuid(train_request.model_id)
    # Hole Modellobjekt per ID (akzeptiert beide Formate)
    model = await get_ai_model_by_id(db, UUID(train_request.model_id))
    model_path = str(Path("data/models") / model.model_tag)
    # Hole Dataset-Pfade aus der DB
    dataset_paths = []
    for ds_id in train_request.dataset_ids:
        if not is_valid_uuid(ds_id):
            raise InvalidDatasetIdError(ds_id)
        ds_uuid = uuid.UUID(ds_id)
        ds = await get_dataset_db(db, ds_uuid)
        # Annahme: file_name ist der relative Pfad im Storage
        dataset_path = Path("data/datasets") / ds.file_name
        dataset_paths.append(str(dataset_path))
    missing = [str(p) for p in dataset_paths if not Path(p).is_file()]
    if missing:
        for p in missing:
            logger.error("Training request failed: Dataset file not found: {}", p)
        raise TrainingDatasetNotFoundError(missing[0])
    logger.info(
        "DPO-Training requested for model_id={}, model_path={}, dataset_paths={}",
        train_request.model_id,
        model_path,
        [str(p) for p in dataset_paths],
    )
    task = TrainingTask(id=uuid4(), task_status=TaskStatus.PENDING)
    await save_training_task(db, task)
    background_tasks.add_task(
        train_model_task,
        db,
        model_path,
        train_request,
        task.id,
        [str(p) for p in dataset_paths],
    )
    logger.info(
        "DPO-Training started in background for model_id={}, task_id={}",
        train_request.model_id,
        str(task.id),
    )
    return Response(status_code=status.HTTP_202_ACCEPTED)


@router.get("/{task_id}/status")
async def get_training_status(
    task_id: UUID,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> TrainingStatusResponse:
    """Gibt den Status und die Metadaten eines Trainingsjobs zurück."""
    task = await get_train_task_by_id(db, task_id)
    if not task:
        raise TrainingTaskNotFoundError(str(task_id))
    return TrainingStatusResponse.from_task(task)
