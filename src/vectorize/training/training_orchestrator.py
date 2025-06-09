"""Orchestrates the end-to-end SBERT training process."""

from pathlib import Path
from uuid import UUID

from loguru import logger
from sentence_transformers import losses
from sqlmodel.ext.asyncio.session import AsyncSession
from torch.utils.data import DataLoader

from vectorize.ai_model.model_source import ModelSource
from vectorize.ai_model.models import AIModel
from vectorize.ai_model.repository import get_ai_model_db, save_ai_model_db
from vectorize.common.task_status import TaskStatus

from .exceptions import DatasetValidationError
from .repository import get_train_task_by_id, update_training_task_status
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
    logger.debug(
        "Starting SBERT training",
        model_path=model_path,
        dataset_paths=dataset_paths,
        output_dir=output_dir,
        task_id=str(task_id),
    )
    model = None
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
        parent_model = await get_ai_model_db(db, train_request.model_tag)
        tag_time = Path(output_dir).name
        new_model_tag = str(Path(output_dir).relative_to("data/models"))
        new_model = AIModel(
            name=f"Fine-tuned: {parent_model.name} {tag_time}",
            model_tag=new_model_tag,
            source=ModelSource.LOCAL,
            trained_from_id=parent_model.id,
            trained_from_tag=parent_model.model_tag,
        )
        new_model_id = await save_ai_model_db(db, new_model)
        task = await get_train_task_by_id(db, task_id)
        if task:
            task.trained_model_id = new_model_id
            await db.commit()
            await db.refresh(task)
        logger.debug(
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
        if model is not None:
            cleanup_resources(model)
        return
    if model is not None:
        cleanup_resources(model)
    await update_training_task_status(db, task_id, TaskStatus.DONE)
    logger.debug(
        "Training finished successfully",
        model_path=model_path,
        task_id=str(task_id),
    )
