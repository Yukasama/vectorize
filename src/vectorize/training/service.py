"""Service layer for training (now SBERT/SentenceTransformer only)."""

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
from sentence_transformers import InputExample, SentenceTransformer, losses
from sqlmodel import Session, create_engine
from sqlmodel.ext.asyncio.session import AsyncSession
from torch.utils.data import DataLoader, Dataset

from vectorize.ai_model.model_source import ModelSource
from vectorize.ai_model.models import AIModel
from vectorize.ai_model.repository import get_ai_model_db, save_ai_model_db
from vectorize.common.task_status import TaskStatus

from .repository import (
    get_train_task_by_id,
    update_training_task_progress,
    update_training_task_status,
)
from .schemas import TrainRequest
from .utils.input_examples import prepare_input_examples
from .utils.safetensors_finder import find_safetensors_file

TRAINING_TIMEOUT_SECONDS = int(
    os.environ.get("TRAINING_TIMEOUT_SECONDS", str(60 * 60 * 2))
)  # 2 hours default
MIN_FREE_DISK_GB = 2


class InputExampleDataset(Dataset):
    """Wraps a list of InputExamples for use with DataLoader.

    Args:
        examples (list): List of InputExample objects.
    """
    def __init__(self, examples: list) -> None:
        """Initializes the dataset.

        Args:
            examples (list): List of InputExample objects.
        """
        self.examples = examples

    def __getitem__(self, idx: int) -> InputExample:
        """Returns the InputExample at the given index.

        Args:
            idx (int): Index of the example.

        Returns:
            InputExample: The example at the given index.
        """
        return self.examples[idx]

    def __len__(self) -> int:
        """Returns the number of examples in the dataset.

        Returns:
            int: Number of examples.
        """
        return len(self.examples)


def _load_sbert_model(model_path: str) -> SentenceTransformer:
    safetensors_path = find_safetensors_file(model_path)
    if safetensors_path:
        model_dir = Path(safetensors_path).parent
        logger.debug(
            "Found .safetensors file for model: {safetensors_path} "
            "(using dir: {model_dir})",
            safetensors_path=safetensors_path,
            model_dir=model_dir,
        )
        return SentenceTransformer(str(model_dir))
    logger.debug(
        "No .safetensors file found, loading model from original path."
    )
    return SentenceTransformer(model_path)


def _prepare_tokenizer(tokenizer) -> None:  # noqa: ANN001
    """Prepare the tokenizer by setting a pad token if needed."""
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})


def _extract_triplet(row: pd.Series) -> list:
    if all(k in row for k in ("Question", "Positive", "Negative")):
        return [row["Question"], row["Positive"], row["Negative"]]
    raise ValueError(
        "Each row must contain the keys 'Question', 'Positive', 'Negative'. "
        f"Found: {list(row.keys())}"
    )


def _prepare_examples(df: pd.DataFrame) -> list[InputExample]:
    return prepare_input_examples(df)


def _get_train_val_examples(
    dataset_paths: list[str],
) -> tuple[list[InputExample], list[InputExample]]:
    train_examples = _prepare_examples(pd.read_json(dataset_paths[0], lines=True))
    if len(dataset_paths) > 1:
        val_examples = _prepare_examples(pd.read_json(dataset_paths[1], lines=True))
    else:
        val_split = int(0.1 * len(train_examples))
        val_examples = train_examples[:val_split]
        train_examples = train_examples[val_split:]
    return train_examples, val_examples


def _save_finetuned_model(model_path: str) -> None:
    """Save the finetuned model metadata to the database."""
    try:
        parent_tag = Path(model_path.rstrip("/")).name
        tag_time = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        new_model_tag = f"{parent_tag}-finetuned-{tag_time}"
        new_model = AIModel(
            name=f"Fine-tuned: {parent_tag} {tag_time}",
            model_tag=new_model_tag,
            source=ModelSource.LOCAL,
            trained_from_tag=parent_tag,
        )
        engine = create_engine("sqlite:///app.db")
        with Session(engine) as session:
            session.add(new_model)
            session.commit()
        logger.info(
            "Finetuned model saved in DB: {new_model_tag} | model_tag={new_model_tag} "
            "| trained_from_tag={parent_tag} | new_model_id={new_model_id}",
            new_model_tag=new_model_tag,
            parent_tag=parent_tag,
            new_model_id=new_model.id,
        )
    except Exception as exc:
        logger.error(
            "Error saving finetuned model to DB: {exc}",
            exc=exc,
        )


def check_disk_space(path: str, min_gb: int = MIN_FREE_DISK_GB) -> bool:
    """Check if the given path has at least min_gb gigabytes free."""
    try:
        _total, _used, free = shutil.disk_usage(path)
        return free >= min_gb * 1024**3
    except Exception as exc:
        logger.warning(
            "Disk space check failed for {path}: {exc}",
            path=path,
            exc=exc,
        )
        return False


class ProgressDBCallback:
    """Callback to update training progress in the database after each batch."""
    def __init__(self, db: AsyncSession, task_id: UUID, total_steps: int) -> None:
        """Initializes the callback."""
        self.db = db
        self.task_id = task_id
        self.total_steps = total_steps
        self.current_step = 0
        self._async_task = None

    def __call__(
        self,
        _score: float | None = None,
        _epoch: int | None = None,
        _steps: float | None = None,
    ) -> None:
        """Updates progress after each batch."""
        self.current_step += 1
        progress = self.current_step / self.total_steps
        loop = asyncio.get_event_loop()
        self._async_task = loop.create_task(
            update_training_task_progress(self.db, self.task_id, progress)
        )


async def train_model_task(
    db: AsyncSession,
    model_path: str,
    train_request: TrainRequest,
    task_id: UUID,
    dataset_paths: list[str],
    output_dir: str,
) -> None:
    """Background task: trains the model, saves new AIModel, updates TrainingTask."""
    logger.debug(
        "Training started for model_path={model_path}, dataset_paths={dataset_paths}, "
        "task_id={task_id}, output_dir={output_dir}",
        model_path=model_path,
        dataset_paths=dataset_paths,
        task_id=task_id,
        output_dir=output_dir,
    )
    await update_training_task_status(db, task_id, TaskStatus.RUNNING)
    parent_dir = Path(output_dir).parent
    if not parent_dir.exists():
        parent_dir.mkdir(parents=True, exist_ok=True)
    if not check_disk_space(str(parent_dir), MIN_FREE_DISK_GB):
        logger.error(
            "Insufficient disk space for training at {parent_dir}. "
            "Minimum required: {min_gb}GB.",
            parent_dir=parent_dir,
            min_gb=MIN_FREE_DISK_GB,
        )
        await update_training_task_status(
            db, task_id, TaskStatus.FAILED, error_msg="Insufficient disk space."
        )
        return
    try:
        if not hasattr(train_request, "model_tag"):
            raise ValueError("TrainRequest must include a model_tag!")
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
                "Training timed out after {timeout} seconds: task_id={task_id}",
                timeout=TRAINING_TIMEOUT_SECONDS,
                task_id=task_id,
            )
            await update_training_task_status(
                db, task_id, TaskStatus.FAILED, error_msg="Training timed out."
            )
            return
        except (OSError, ValueError, RuntimeError) as exc:
            logger.error(
                "Training failed due to system/runtime error: {exc}",
                exc=exc,
            )
            await update_training_task_status(
                db, task_id, TaskStatus.FAILED, error_msg=str(exc)
            )
            return
        except Exception as exc:
            logger.error(
                "Unexpected error during training: {exc}",
                exc=exc,
            )
            await update_training_task_status(
                db, task_id, TaskStatus.FAILED, error_msg=str(exc)
            )
            return
        task = await get_train_task_by_id(db, task_id)
        if task and task.task_status == TaskStatus.CANCELED:
            logger.debug(
                "Training was canceled, skipping model save and DB entry for "
                "task_id={task_id}",
                task_id=task_id,
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
            "Saved new finetuned model in DB: {new_model_name} | "
            "model_tag={new_model_tag} | trained_from_tag={orig_model_tag} | "
            "new_model_id={new_model_id}",
            new_model_name=new_model.name,
            new_model_tag=new_model_tag,
            orig_model_tag=orig_model.model_tag,
            new_model_id=new_model_id,
        )
        task = await get_train_task_by_id(db, task_id)
        if task:
            task.trained_model_id = new_model_id
            await db.commit()
            await db.refresh(task)
        await update_training_task_status(db, task_id, TaskStatus.DONE)
        logger.debug(
            "Training finished successfully for model_path={model_path}, "
            "task_id={task_id}, new_model_id={new_model_id}",
            model_path=model_path,
            task_id=task_id,
            new_model_id=new_model_id,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        logger.error(
            "Training failed due to system/runtime error: {exc}",
            exc=exc,
        )
        await update_training_task_status(
            db, task_id, TaskStatus.FAILED, error_msg=str(exc)
        )
    except Exception as exc:
        logger.exception(
            "Training failed: task_id={task_id} - {exc}",
            task_id=task_id,
            exc=exc,
        )
        await update_training_task_status(
            db, task_id, TaskStatus.FAILED, error_msg=str(exc)
        )
    finally:
        try:
            del orig_model
        except Exception as exc:
            logger.warning(
                "Cleanup failed for orig_model: {exc}",
                exc=exc,
            )
        try:
            gc.collect()
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            logger.warning(
                "Cleanup failed (GC/CUDA): {exc}",
                exc=exc,
            )


async def _run_training_with_progress(
    db: AsyncSession,
    model_path: str,
    train_request: TrainRequest,
    task_id: UUID,
    dataset_paths: list[str],
    output_dir: str,
) -> None:
    """Run SBERT triplet training and update progress after each batch.

    Args:
        db: The database session.
        model_path: Path to the base model.
        train_request: Training configuration.
        task_id: The training task ID.
        dataset_paths: List of dataset file paths.
        output_dir: Output directory for the trained model.
    """
    dataset_file = dataset_paths[0]
    safetensors_path = find_safetensors_file(model_path)
    if safetensors_path:
        model_dir = Path(safetensors_path).parent
        logger.debug(
            "Found .safetensors file for model: {safetensors_path} "
            "(using dir: {model_dir})",
            safetensors_path=safetensors_path,
            model_dir=model_dir,
        )
        model = SentenceTransformer(str(model_dir))
    else:
        logger.debug(
            "No .safetensors file found, loading model from original path."
        )
        model = SentenceTransformer(model_path)
    df = pd.read_json(dataset_file, lines=True)
    train_examples = prepare_input_examples(df)
    if len(dataset_paths) > 1:
        # Only used for validation, not needed here
        _ = pd.read_json(dataset_paths[1], lines=True)
    else:
        val_split = int(0.1 * len(train_examples))
        # val_examples = train_examples[:val_split]  # unused
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
    logger.debug(
        "SBERT training complete. Model saved at: {output_dir_path}",
        output_dir_path=output_dir_path,
    )
    try:
        del model
    except Exception as exc:
        logger.warning(
            "Cleanup failed (model): {exc}",
            exc=exc,
        )
    try:
        gc.collect()
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        logger.warning(
            "Cleanup failed (GC/CUDA): {exc}",
            exc=exc,
        )


def train_sbert_triplet_service(
    model_path: str,
    train_request: TrainRequest,
    dataset_paths: list[str],
    output_dir: str,
) -> None:
    """Train a SentenceTransformer (SBERT) model with CosineSimilarityLoss.

    Args:
        model_path (str): Path to the base model.
        train_request (TrainRequest): Training configuration.
        dataset_paths (list[str]): List of dataset file paths.
        output_dir (str): Output directory for the trained model.
    """
    model = _load_sbert_model(model_path)
    _prepare_tokenizer(model.tokenizer)
    train_examples, _val_examples = _get_train_val_examples(dataset_paths)
    train_dataset = InputExampleDataset(train_examples)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_request.per_device_train_batch_size,
        shuffle=True,
        num_workers=getattr(train_request, "dataloader_num_workers", 0) or 0,
    )
    loss = losses.CosineSimilarityLoss(model)
    fit_kwargs = {
        "epochs": train_request.epochs,
        "warmup_steps": train_request.warmup_steps or 0,
        "show_progress_bar": (
            train_request.show_progress_bar
            if getattr(train_request, "show_progress_bar", None) is not None
            else True
        ),
        "output_path": output_dir,
    }
    for key in [
        "optimizer_name", "scheduler", "weight_decay", "max_grad_norm", "use_amp",
        "evaluation_steps", "save_best_model", "save_each_epoch",
        "save_optimizer_state", "device"
    ]:
        value = getattr(train_request, key, None)
        if value is not None:
            fit_kwargs[key] = value
    logger.info(
        "Starting SBERT contrastive training with parameters: {fit_kwargs} | "
        "Model dir: {model_path} | Output: {output_dir}",
        fit_kwargs=fit_kwargs,
        model_path=model_path,
        output_dir=output_dir,
    )
    model.fit(
        train_objectives=[(train_dataloader, loss)],
        **fit_kwargs,
    )
    model.save(output_dir)
    logger.info(
        "SBERT model saved at {output_dir} (trained from: {model_path})",
        output_dir=output_dir,
        model_path=model_path,
    )
    _save_finetuned_model(model_path)
    try:
        del model
    except Exception as exc:
        logger.warning(
            "Cleanup failed (model): {exc}",
            exc=exc,
        )
    gc.collect()
