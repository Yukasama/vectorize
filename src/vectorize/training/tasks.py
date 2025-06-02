"""Background task for model training."""

import asyncio
import gc
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import torch
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
from .utils.uuid_validator import is_valid_uuid

TRAINING_TIMEOUT_SECONDS = int(
    os.environ.get("TRAINING_TIMEOUT_SECONDS", str(60 * 60 * 2))
)  # 2 hours default

MIN_FREE_DISK_GB = 2  # Minimum free disk space in GB required for training


def check_disk_space(path: str, min_gb: int = MIN_FREE_DISK_GB) -> bool:
    """Check if the given path has at least min_gb gigabytes free."""
    try:
        _total, _used, free = shutil.disk_usage(path)
        return free >= min_gb * 1024**3
    except Exception as exc:
        logger.warning(f"Disk space check failed for {path}: {exc}")
        return False


async def _is_task_canceled(db: AsyncSession, task_id: UUID) -> bool:
    task = await get_train_task_by_id(db, task_id)
    return task is not None and task.task_status == TaskStatus.CANCELED


# NOTE: Argument count exceeds 5 due to business logic and clarity requirements.
# This is an accepted exception for these training orchestration functions.
async def train_model_task(  # noqa: PLR0913,PLR0917, PLR0912, PLR0915
    db: AsyncSession,
    model_path: str,
    train_request: TrainRequest,
    task_id: UUID,
    dataset_paths: list[str],
    output_dir: str,
) -> None:
    """Background task: trains the model, saves new AIModel, updates TrainingTask."""
    logger.debug(
        "Training started for model_path=%s, dataset_paths=%s, task_id=%s, "
        "output_dir=%s",
        model_path,
        dataset_paths,
        task_id,
        output_dir,
    )
    await update_training_task_status(db, task_id, TaskStatus.RUNNING)
    parent_dir = Path(output_dir).parent
    if not parent_dir.exists():
        parent_dir.mkdir(parents=True, exist_ok=True)
    if not check_disk_space(str(parent_dir), MIN_FREE_DISK_GB):
        logger.error(
            f"Insufficient disk space for training at {parent_dir}. Minimum "
            f"required: {MIN_FREE_DISK_GB}GB."
        )
        await update_training_task_status(
            db, task_id, TaskStatus.FAILED, error_msg="Insufficient disk space."
        )
        return
    try:
        if not is_valid_uuid(train_request.model_id):
            raise InvalidModelIdError(train_request.model_id)
        norm_model_id = UUID(train_request.model_id)
        orig_model = await get_ai_model_by_id(db, norm_model_id)

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
        except (OSError, ValueError, RuntimeError) as exc:
            logger.error(f"Training failed due to system/runtime error: {exc}")
            await update_training_task_status(
                db, task_id, TaskStatus.FAILED, error_msg=str(exc)
            )
            return
        except Exception as exc:
            logger.error(f"Unexpected error during training: {exc}")
            await update_training_task_status(
                db, task_id, TaskStatus.FAILED, error_msg=str(exc)
            )
            return

        task = await get_train_task_by_id(db, task_id)
        if task and task.task_status == TaskStatus.CANCELED:
            logger.debug(
                "Training was canceled, skipping model save and DB entry for "
                "task_id=%s",
                task_id,
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
        logger.debug(
            f"Training finished successfully for model_path={model_path}, "
            f"task_id={task_id}, new_model_id={new_model_id}"
        )
    except (OSError, ValueError, RuntimeError) as exc:
        logger.error(f"Training failed due to system/runtime error: {exc}")
        await update_training_task_status(
            db, task_id, TaskStatus.FAILED, error_msg=str(exc)
        )
    except Exception as exc:
        logger.exception(f"Training failed: task_id={task_id} - {exc}")
        await update_training_task_status(
            db, task_id, TaskStatus.FAILED, error_msg=str(exc)
        )
    finally:
        try:
            del orig_model
        except Exception as exc:
            logger.warning(f"Cleanup failed for orig_model: {exc}")
        try:
            gc.collect()
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            logger.warning(f"Cleanup failed (GC/CUDA): {exc}")


# NOTE: Progress is updated after each epoch.
# For batch-level progress, a custom callback is needed.
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
    """Run DPO training and update progress after each epoch (epoch-based progress)."""
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
        if await _is_task_canceled(db, task_id):
            logger.debug(
                "Training canceled at epoch %s/%s", epoch + 1, train_request.epochs
            )
            await update_training_task_status(
                db, task_id, TaskStatus.CANCELED, error_msg="Training canceled by user"
            )
            return
        trainer.train(resume_from_checkpoint=None)
        progress = ((epoch + 1) * steps_per_epoch) / total_steps
        await update_training_task_progress(db, task_id, progress)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir_path))
    tokenizer.save_pretrained(str(output_dir_path))
    logger.debug("DPO training complete. Model saved at: %s", output_dir_path)
    try:
        del model
        del tokenizer
    except Exception as exc:
        logger.warning(f"Cleanup failed (model/tokenizer): {exc}")
    try:
        gc.collect()
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        logger.warning(f"Cleanup failed (GC/CUDA): {exc}")
