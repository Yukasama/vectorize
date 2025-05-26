"""Training router."""

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Response, status
from loguru import logger

from .exceptions import TrainingDatasetNotFoundError
from .schemas import TrainRequest
from .tasks import train_model_task

__all__ = ["router"]

router = APIRouter(tags=["Training"])


@router.post("/train")
async def train_model(
    train_request: TrainRequest,
    background_tasks: BackgroundTasks,
) -> Response:
    """Start model training as a background task."""
    for dataset_path in train_request.dataset_paths:
        path = Path(dataset_path)
        if not path.is_file():
            logger.error(
                "Training request failed: Dataset file not found: %s",
                dataset_path,
            )
            raise TrainingDatasetNotFoundError(dataset_path)
    logger.info(
        "Training requested for model_path=%s, dataset_paths=%s",
        train_request.model_path,
        train_request.dataset_paths,
    )
    background_tasks.add_task(train_model_task, train_request)
    logger.info(
        "Training started in background for model_path=%s",
        train_request.model_path,
    )
    return Response(
        content=(
            "{" +
            '"message": "Training started", ' +
            '"model_path": "' + train_request.model_path + '"' +
            "}"
        ),
        status_code=status.HTTP_202_ACCEPTED,
        media_type="application/json",
    )
