"""Background task for model training."""

from loguru import logger

from .exceptions import TrainingModelNotFoundError
from .schemas import TrainRequest
from .service import train_model_service_svc


def train_model_task(train_request: TrainRequest) -> None:
    """Wrapper for the background task, calls the training service function."""
    logger.info(
        "[BG] Training started for model_path=%s, dataset_paths=%s",
        train_request.model_path,
        train_request.dataset_paths,
    )
    try:
        train_model_service_svc(train_request)
        logger.info(
            "[BG] Training finished successfully for model_path=%s",
            train_request.model_path,
        )
    except TrainingModelNotFoundError:
        logger.error(
            "[BG] Training failed: Invalid model path: %s",
            train_request.model_path,
        )
    except Exception as e:
        logger.error(
            "[BG] Training failed for model_path=%s: %s",
            train_request.model_path,
            str(e),
        )
        raise
