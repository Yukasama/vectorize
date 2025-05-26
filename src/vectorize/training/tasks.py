"""Background-Task für das Modelltraining."""

from loguru import logger

from .schemas import TrainRequest
from .service import train_model_service_svc


def train_model_task(train_request: TrainRequest) -> None:
    """Wrapper für die Hintergrund-Task, ruft die Trainings-Servicefunktion auf."""
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
    except Exception as e:
        logger.error(
            "[BG] Training failed for model_path=%s: %s",
            train_request.model_path,
            str(e),
        )
        raise
