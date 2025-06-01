"""Trains a model with DPOTrainer on prompt/chosen/rejected data."""

from pathlib import Path

from datasets import load_dataset
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from .schemas import TrainRequest


def train_model_service_svc(
    model_path: str,
    train_request: TrainRequest,
    dataset_paths: list[str],
) -> None:
    """Trains a model with DPOTrainer on prompt/chosen/rejected data."""
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
    trainer.train()
    output_dir = Path(train_request.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info(f"DPO-Training abgeschlossen. Modell gespeichert unter: {output_dir}")
