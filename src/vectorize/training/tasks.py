"""Background-Task für das Modelltraining."""

from loguru import logger

from .schemas import TrainRequest
from .service import train_model_service


def train_model_task(train_request: TrainRequest) -> None:
    """Wrapper für die Hintergrund-Task, ruft die Trainings-Servicefunktion auf."""
    logger.info(
        "[BG] Training started for model_tag=%s, dataset_path=%s",
        train_request.model_tag,
        train_request.dataset_path,
    )
    try:
        train_model_service(train_request)
        logger.info(
            "[BG] Training finished successfully for model_tag=%s",
            train_request.model_tag,
        )
    except Exception as e:
        logger.error(
            "[BG] Training failed for model_tag=%s: %s",
            train_request.model_tag,
            str(e),
        )
        raise
