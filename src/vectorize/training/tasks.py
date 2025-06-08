"""Background task for model training."""

import asyncio
import gc
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pandas as pd
import torch
from loguru import logger
from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    losses,
)
from sqlmodel.ext.asyncio.session import AsyncSession
from torch.utils.data import DataLoader

from vectorize.ai_model.model_source import ModelSource
from vectorize.ai_model.models import AIModel
from vectorize.ai_model.repository import get_ai_model_db, save_ai_model_db
from vectorize.common.task_status import TaskStatus

from .exceptions import InvalidModelIdError
from .repository import (
    get_train_task_by_id,
    update_training_task_progress,
    update_training_task_status,
)
from .schemas import TrainRequest
from .service import InputExampleDataset
from .utils.safetensors_finder import find_safetensors_file

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
        if not hasattr(train_request, "model_tag"):
            raise InvalidModelIdError(
                "TrainRequest must include a model_tag!"
            )
        orig_model = await get_ai_model_db(db, train_request.model_tag)

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
            trained_from_id=None,
            trained_from_tag=orig_model.model_tag,
        )
        new_model_id = await save_ai_model_db(db, new_model)
        logger.info(
            "Saved new finetuned model in DB: %s | model_tag=%s | "
            "trained_from_tag=%s | new_model_id=%s",
            new_model.name,
            new_model_tag,
            orig_model.model_tag,
            new_model_id,
        )
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


class ProgressDBCallback:
    """Callback to update training progress in the database after each batch.

    Args:
        db (AsyncSession): The database session.
        task_id (UUID): The training task ID.
        total_steps (int): The total number of steps in training.
    """
    def __init__(self, db: AsyncSession, task_id: UUID, total_steps: int) -> None:
        """Initializes the callback.

        Args:
            db (AsyncSession): The database session.
            task_id (UUID): The training task ID.
            total_steps (int): The total number of steps in training.
        """
        self.db = db
        self.task_id = task_id
        self.total_steps = total_steps
        self.current_step = 0
        self._async_task = None  # Store reference to avoid RUF006

    def __call__(
        self,
        _score: float | None = None,
        _epoch: int | None = None,
        _steps: float | None = None,
    ) -> None:
        """Updates progress after each batch.

        Args:
            score (float, optional): The current score (unused).
            epoch (int, optional): The current epoch (unused).
            _steps (float, optional): The current step (unused, intentionally ignored).
        """
        self.current_step += 1
        progress = self.current_step / self.total_steps
        loop = asyncio.get_event_loop()
        self._async_task = loop.create_task(
            update_training_task_progress(self.db, self.task_id, progress)
        )


# NOTE: Progress is updated after each epoch.
# For batch-level progress, a custom callback is needed.
# NOTE: Argument count exceeds 5 due to business logic and clarity requirements.
# This is an accepted exception for these training orchestration functions.
async def _run_training_with_progress(  # noqa: PLR0913, PLR0917, PLR0914, RUF029
    db: AsyncSession,
    model_path: str,
    train_request: TrainRequest,
    task_id: UUID,
    dataset_paths: list[str],
    output_dir: str,
) -> None:
    """Run SBERT triplet training and update progress after each batch using callback.

    Args:
        db (AsyncSession): The database session.
        model_path (str): Path to the base model.
        train_request (TrainRequest): Training configuration.
        task_id (UUID): The training task ID.
        dataset_paths (list[str]): List of dataset file paths.
        output_dir (str): Output directory for the trained model.
    """
    dataset_file = dataset_paths[0]
    safetensors_path = find_safetensors_file(model_path)
    if safetensors_path:
        model_dir = Path(safetensors_path).parent
        logger.debug(
            "Found .safetensors file for model: %s (using dir: %s)",
            safetensors_path,
            model_dir,
        )
        model = SentenceTransformer(str(model_dir))
    else:
        logger.debug("No .safetensors file found, loading model from original path.")
        model = SentenceTransformer(model_path)
    df = pd.read_json(dataset_file, lines=True)
    train_examples = []
    for _, row in df.iterrows():
        q, pos, neg = row["Question"], row["Positive"], row["Negative"]
        train_examples.extend([
            InputExample(texts=[q, pos], label=1.0),
            InputExample(texts=[q, neg], label=-1.0),
        ])
    if len(dataset_paths) > 1:
        val_df = pd.read_json(dataset_paths[1], lines=True)
        val_examples = []
        for _, row in val_df.iterrows():
            q, pos, neg = row["Question"], row["Positive"], row["Negative"]
            val_examples.extend([
                InputExample(texts=[q, pos], label=1.0),
                InputExample(texts=[q, neg], label=-1.0),
            ])
    else:
        val_split = int(0.1 * len(train_examples))
        val_examples = train_examples[:val_split]
        train_examples = train_examples[val_split:]
    train_dataset = InputExampleDataset(train_examples)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_request.per_device_train_batch_size,
        shuffle=True,
        collate_fn=lambda x: x,
    )
    loss = losses.CosineSimilarityLoss(model)
    num_epochs = train_request.epochs
    steps_per_epoch = len(train_dataloader)
    total_steps = num_epochs * steps_per_epoch
    progress_callback = ProgressDBCallback(db, task_id, total_steps)
    model.fit(
        train_objectives=[(train_dataloader, loss)],
        epochs=num_epochs,
        warmup_steps=train_request.warmup_steps or 0,
        show_progress_bar=False,
        output_path=str(Path(output_dir)),
        callback=progress_callback,
    )
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir_path))
    logger.debug("SBERT training complete. Model saved at: %s", output_dir_path)
    try:
        del model
    except Exception as exc:
        logger.warning("Cleanup failed (model): %s", exc)
    try:
        gc.collect()
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        logger.warning("Cleanup failed (GC/CUDA): %s", exc)
