"""Trains a model with DPOTrainer on prompt/chosen/rejected data."""

from pathlib import Path
from typing import Optional

from datasets import load_dataset
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from .schemas import TrainRequest


class ProgressCallback:
    """Callback to update training progress in the database."""
    def __init__(self, total_steps: int, on_update):
        self.total_steps = max(total_steps, 1)
        self.on_update = on_update
        self.current_step = 0

    def __call__(self, step: Optional[int] = None):
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
    output_dir: str,  # output_dir als Funktionsargument
    progress_callback=None,
) -> None:
    """Trains a model with DPOTrainer on prompt/chosen/rejected data. Optionally tracks progress."""
    logger.info("Starte DPO-Training mit Hugging Face TRL.")
    if not dataset_paths:
        raise ValueError("Keine Datensätze angegeben.")
    # Für DPO: Wir nehmen den ersten Datensatz (kann ggf. erweitert werden)
    dataset_file = dataset_paths[0]
    if not Path(dataset_file).is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    dataset = load_dataset("json", data_files=dataset_file, split="train")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = DPOConfig(
        learning_rate=train_request.learning_rate,
        per_device_train_batch_size=train_request.per_device_train_batch_size,
        num_train_epochs=train_request.epochs,
    )
    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,  # type: ignore
        processing_class=tokenizer,
    )
    try:
        # Robust step count: use len() if possible, else fallback to 1
        try:
            steps_per_epoch = len(dataset)  # type: ignore
        except Exception:
            steps_per_epoch = 1
        total_steps = train_request.epochs * steps_per_epoch
        if progress_callback:
            for epoch in range(train_request.epochs):
                trainer.train(resume_from_checkpoint=None)
                progress_callback((epoch + 1) * steps_per_epoch)
        else:
            trainer.train()
    finally:
        # output_dir wird jetzt immer automatisch als str übergeben
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_dir_path))
        tokenizer.save_pretrained(str(output_dir_path))
        logger.info(f"DPO-Training abgeschlossen. Modell gespeichert unter: {output_dir_path}")
        # Speicherbereinigung nach dem Training
        try:
            del model
            del tokenizer
        except Exception:
            pass
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
