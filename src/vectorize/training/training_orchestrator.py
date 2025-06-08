"""Orchestrates the end-to-end SBERT training process."""

from pathlib import Path
from uuid import UUID

from loguru import logger
from sentence_transformers import losses
from sqlmodel.ext.asyncio.session import AsyncSession
from torch.utils.data import DataLoader

from vectorize.common.task_status import TaskStatus

from .exceptions import DatasetValidationError
from .repository import update_training_task_status
from .schemas import TrainRequest
from .utils.cleanup import cleanup_resources
from .utils.input_examples import InputExampleDataset, prepare_input_examples
from .utils.model_loader import load_and_prepare_model
from .utils.validators import TrainingDataValidator


async def run_training(  # noqa: PLR0913, PLR0917
    db: AsyncSession,
    model_path: str,
    train_request: TrainRequest,
    task_id: UUID,
    dataset_paths: list[str],
    output_dir: str,
) -> None:
    """Main orchestration function for SBERT training.

    Args:
        db (AsyncSession): Database session.
        model_path (str): Path to the base model.
        train_request (TrainRequest): Training configuration.
        task_id (UUID): Training task ID.
        dataset_paths (list[str]): List of dataset file paths.
        output_dir (str): Output directory for the trained model.
    """
    logger.info(
        "Starting SBERT training",
        model_path=model_path,
        dataset_paths=dataset_paths,
        output_dir=output_dir,
        task_id=str(task_id),
    )
    try:
        model = load_and_prepare_model(model_path)
        df = TrainingDataValidator.validate_dataset(Path(dataset_paths[0]))
        train_examples = prepare_input_examples(df)
        if len(dataset_paths) > 1:
            val_df = TrainingDataValidator.validate_dataset(Path(dataset_paths[1]))
            _ = prepare_input_examples(val_df)
        else:
            val_split = int(0.1 * len(train_examples))
            train_examples = train_examples[val_split:]
        train_dataset = InputExampleDataset(train_examples)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_request.per_device_train_batch_size,
        )
        loss = losses.CosineSimilarityLoss(model)
        model.fit(
            train_objectives=[(train_dataloader, loss)],
            epochs=train_request.epochs,
            warmup_steps=train_request.warmup_steps or 0,
            show_progress_bar=False,
            output_path=str(Path(output_dir)),
        )
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model.save(str(output_dir))
        logger.info(
            "SBERT training complete. Model saved at: {}",
            output_dir,
        )
    except (OSError, RuntimeError, DatasetValidationError) as exc:
        logger.error(
            "Training failed",
            task_id=str(task_id),
            exc=str(exc),
        )
        await update_training_task_status(
            db,
            task_id,
            TaskStatus.FAILED,
            error_msg=str(exc),
        )
        cleanup_resources(model)
        return
    cleanup_resources(model)
    await update_training_task_status(db, task_id, TaskStatus.DONE)
    logger.info(
        "Training finished successfully",
        model_path=model_path,
        task_id=str(task_id),
    )
