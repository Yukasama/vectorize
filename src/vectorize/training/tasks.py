"""Background task for model training."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from datasets import load_dataset
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from vectorize.ai_model.model_source import ModelSource
from vectorize.ai_model.models import AIModel
from vectorize.ai_model.repository import get_ai_model_by_id, save_ai_model_db
from vectorize.common.task_status import TaskStatus

from .exceptions import InvalidModelIdError
from .repository import (
    get_train_task_by_id,
    update_training_task_progress,
    update_training_task_status,
)
from .schemas import TrainRequest
from .utils.uuid_utils import is_valid_uuid

TRAINING_TIMEOUT_SECONDS = 60 * 60 * 2  # 2 Stunden Timeout


# NOTE: Argument count exceeds 5 due to business logic and clarity requirements.
# This is an accepted exception for these training orchestration functions.
async def train_model_task(  # noqa: PLR0913,PLR0917
    db: AsyncSession,
    model_path: str,
    train_request: TrainRequest,
    task_id: UUID,
    dataset_paths: list[str],
    output_dir: str,
) -> None:
    """Background task: trains the model, saves new AIModel, updates TrainingTask."""
    logger.info(
        "Training started for model_path=%s, dataset_paths=%s, task_id=%s, "
        "output_dir=%s",
        model_path,
        dataset_paths,
        task_id,
        output_dir,
    )
    try:
        # Validate and normalize model ID (defensive)
        if not is_valid_uuid(train_request.model_id):
            raise InvalidModelIdError(train_request.model_id)
        norm_model_id = UUID(train_request.model_id)
        orig_model = await get_ai_model_by_id(db, norm_model_id)

        # Training with progress update after each epoch, now with timeout
        try:
            await asyncio.wait_for(
                _run_training_with_progress(
                    db=db,
                    model_path=model_path,
                    train_request=train_request,
                    task_id=task_id,
                    dataset_paths=dataset_paths,
                    output_dir=output_dir,
                ),
                timeout=TRAINING_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            logger.error(
                "Training timed out after %s seconds: task_id=%s",
                TRAINING_TIMEOUT_SECONDS,
                task_id,
            )
            await update_training_task_status(
                db, task_id, TaskStatus.FAILED, error_msg="Training timed out."
            )
            return

        tag_time = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        new_model_tag = f"{orig_model.model_tag}-finetuned-{tag_time}"
        new_model = AIModel(
            name=f"Fine-tuned: {orig_model.name} {tag_time}",
            model_tag=new_model_tag,
            source=ModelSource.LOCAL,
            trained_from_id=orig_model.id,
        )
        new_model_id = await save_ai_model_db(db, new_model)
        task = await get_train_task_by_id(db, task_id)
        if task:
            task.trained_model_id = new_model_id
            await db.commit()
            await db.refresh(task)
        await update_training_task_status(db, task_id, TaskStatus.DONE)
        logger.info(
            f"Training finished successfully for model_path={model_path}, "
            f"task_id={task_id}, new_model_id={new_model_id}"
        )
    except Exception as exc:
        logger.exception(f"Training failed: task_id={task_id} - {exc}")
        await update_training_task_status(
            db, task_id, TaskStatus.FAILED, error_msg=str(exc)
        )


# NOTE: Argument count exceeds 5 due to business logic and clarity requirements.
# This is an accepted exception for these training orchestration functions.
async def _run_training_with_progress(  # noqa: PLR0913,PLR0917
    db: AsyncSession,
    model_path: str,
    train_request: TrainRequest,
    task_id: UUID,
    dataset_paths: list[str],
    output_dir: str,
) -> None:
    """Run DPO training and update progress after each epoch."""
    dataset_file = dataset_paths[0]
    dataset = load_dataset("json", data_files=dataset_file, split="train")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = DPOConfig(
        learning_rate=train_request.learning_rate,
        per_device_train_batch_size=train_request.per_device_train_batch_size,
        num_train_epochs=1,
        output_dir=output_dir,  # Enforce output_dir for all trainer artifacts
    )
    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,  # type: ignore
        processing_class=tokenizer,
    )
    steps_per_epoch = 1
    try:
        steps_per_epoch = len(dataset)  # type: ignore
        if not isinstance(steps_per_epoch, int) or steps_per_epoch <= 0:
            steps_per_epoch = 1
    except Exception:
        steps_per_epoch = 1
    total_steps = train_request.epochs * steps_per_epoch
    for epoch in range(train_request.epochs):
        trainer.train(resume_from_checkpoint=None)
        progress = ((epoch + 1) * steps_per_epoch) / total_steps
        await update_training_task_progress(db, task_id, progress)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir_path))
    tokenizer.save_pretrained(str(output_dir_path))
    logger.info(
        "DPO training complete. Model saved at: %s", output_dir_path
    )
