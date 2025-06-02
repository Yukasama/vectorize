"""Trains a model with DPOTrainer on prompt/chosen/rejected data."""

import gc
from collections.abc import Callable
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from datasets import load_dataset
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from .schemas import TrainRequest


class ProgressCallback:
    """Callback to update training progress in the database."""

    def __init__(self, total_steps: int, on_update: Callable[[float], None]) -> None:
        """Initialize ProgressCallback.

        Args:
            total_steps (int): Total number of steps for training.
            on_update (Callable[[float], None]): Callback to update progress.
        """
        self.total_steps = max(total_steps, 1)
        self.on_update = on_update
        self.current_step = 0

    def __call__(self, step: int | None = None) -> None:
        """Update progress.

        Args:
            step (int | None): Current step number.
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        progress = min(self.current_step / self.total_steps, 1.0)
        self.on_update(progress)


def train_model_service_svc(
    model_path: str,
    train_request: TrainRequest,
    dataset_paths: list[str],
    output_dir: str,
    progress_callback: Callable[[int], None] | None = None,
) -> None:
    """Train a model with DPOTrainer on prompt/chosen/rejected data.

    Args:
        model_path (str): Path to the model.
        train_request (TrainRequest): Training request parameters.
        dataset_paths (list[str]): List of dataset file paths.
        output_dir (str): Output directory for the trained model.
        progress_callback (Callable[[int], None] | None): Optional callback for
            progress updates.
    """
    logger.debug("Starting DPO training with Hugging Face TRL.")
    if not dataset_paths:
        raise ValueError("No datasets provided.")
    dataset_file = Path(dataset_paths[0])
    if not dataset_file.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    dataset = load_dataset("json", data_files=str(dataset_file), split="train")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dpo_config_kwargs = train_request.model_dump(exclude_unset=True)
    dpo_config_kwargs["num_train_epochs"] = dpo_config_kwargs.pop("epochs")
    dpo_config_kwargs.pop("model_id", None)
    dpo_config_kwargs.pop("dataset_ids", None)
    config = DPOConfig(**dpo_config_kwargs)
    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,  # type: ignore
        processing_class=tokenizer,
    )
    try:
        try:
            steps_per_epoch = len(dataset)  # type: ignore
        except Exception:
            steps_per_epoch = 1
        if progress_callback:
            for epoch in range(train_request.epochs):
                trainer.train(resume_from_checkpoint=None)
                progress_callback((epoch + 1) * steps_per_epoch)
        else:
            trainer.train()
    finally:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_dir_path))
        tokenizer.save_pretrained(str(output_dir_path))
        logger.debug(f"DPO training finished. Model saved at: {output_dir_path}")
        try:
            del model
            del tokenizer
        except Exception as exc:
            logger.warning(f"Cleanup failed: {exc}")
        gc.collect()
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
