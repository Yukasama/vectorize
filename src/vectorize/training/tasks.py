"""Background task for model training."""

from datetime import UTC, datetime
from uuid import UUID

from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

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
from .service import train_model_service_svc
from .utils.uuid_utils import is_valid_uuid


async def train_model_task(
    db: AsyncSession,
    model_path: str,
    train_request: TrainRequest,
    task_id: UUID,
    dataset_paths: list[str],
) -> None:
    """Background task: trains the model, saves new AIModel, updates TrainingTask."""
    logger.info(
        "Training started for model_path={}, dataset_paths={}, task_id={}",
        model_path,
        dataset_paths,
        task_id,
    )
    try:
        # Validierung und Normalisierung der Modell-ID (defensiv)
        if not is_valid_uuid(train_request.model_id):
            raise InvalidModelIdError(train_request.model_id)
        norm_model_id = UUID(train_request.model_id)
        orig_model = await get_ai_model_by_id(db, norm_model_id)

        # Training epochweise, Fortschritt nach jeder Epoche synchron updaten
        from vectorize.training.service import train_model_service_svc
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
        from pathlib import Path

        dataset_file = dataset_paths[0]
        dataset = load_dataset("json", data_files=dataset_file, split="train")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        config = DPOConfig(
            learning_rate=train_request.learning_rate,
            per_device_train_batch_size=train_request.per_device_train_batch_size,
            num_train_epochs=1,  # Wir steuern die Epochen selbst
        )
        trainer = DPOTrainer(
            model=model,
            args=config,
            train_dataset=dataset,  # type: ignore
            processing_class=tokenizer,
        )
        # Robust step count: use length for datasets.Dataset, fallback to 1 for IterableDataset
        steps_per_epoch = 1
        try:
            # Hugging Face datasets.Dataset has __len__
            steps_per_epoch = int(getattr(dataset, '__len__', lambda: 1)())
        except Exception:
            steps_per_epoch = 1
        if not isinstance(steps_per_epoch, int):
            steps_per_epoch = 1
        total_steps = train_request.epochs * steps_per_epoch
        for epoch in range(train_request.epochs):
            trainer.train(resume_from_checkpoint=None)
            progress = ((epoch + 1) * steps_per_epoch) / total_steps
            await update_training_task_progress(db, task_id, progress)
        output_dir = Path(train_request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        logger.info(f"DPO-Training abgeschlossen. Modell gespeichert unter: {output_dir}")

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
            "Training finished successfully for model_path={}, task_id={}, "
            "new_model_id={}",
            model_path,
            task_id,
            new_model_id,
        )
    except Exception as exc:
        logger.exception(f"Training failed: task_id={task_id} - {exc}")
        await update_training_task_status(
            db, task_id, TaskStatus.FAILED, error_msg=str(exc)
        )
