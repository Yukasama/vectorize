"""Service for model training (Transformers Trainer API)."""

from pathlib import Path

from datasets import load_dataset
from loguru import logger
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from vectorize.config import settings

from .exceptions import TrainingDatasetNotFoundError
from .schemas import TrainRequest


def _find_hf_model_dir_svc(base_path: Path) -> Path:
    """Suche rekursiv nach einem Unterordner mit config.json (Huggingface-Format)."""
    if (base_path / "config.json").is_file():
        return base_path
    for subdir in base_path.rglob(""):
        if (subdir / "config.json").is_file():
            return subdir
    raise FileNotFoundError(f"Kein Unterordner mit config.json in {base_path}")


def train_model_service_svc(train_request: TrainRequest) -> None:
    """Training logic for local or Huggingface models.

    Loads model and dataset, starts training with Huggingface Trainer.
    Saves trained models under data/models/trained_models.
    """
    # Zielverzeichnis: data/models/trained_models/<model_tag>-finetuned
    base_dir = settings.model_upload_dir / "trained_models"
    base_dir.mkdir(parents=True, exist_ok=True)
    model_dir_name = f"{Path(train_request.model_tag).name}-finetuned"
    output_dir = base_dir / model_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if dataset file exists before loading
    dataset_path = Path(train_request.dataset_path)
    if not dataset_path.is_file():
        logger.error(
            "Training failed: Dataset file not found: %s",
            train_request.dataset_path,
        )
        raise TrainingDatasetNotFoundError(train_request.dataset_path)

    # Prüfe, ob model_tag ein existierender Ordner ist (lokales Modell)
    model_path = Path(train_request.model_tag)
    if model_path.exists() and model_path.is_dir():
        try:
            model_load_path = str(_find_hf_model_dir_svc(model_path).resolve())
        except FileNotFoundError as e:
            logger.error(str(e))
            raise TrainingDatasetNotFoundError(
                f"Kein Huggingface-Modelldir gefunden: {e}"
            ) from e
    else:
        model_load_path = train_request.model_tag  # Huggingface Hub Name

    # Load model and tokenizer (local or Huggingface)
    model = AutoModelForSequenceClassification.from_pretrained(model_load_path)
    tokenizer = AutoTokenizer.from_pretrained(model_load_path)

    # Load dataset (CSV, expects columns: text & label)
    data_files = {"train": train_request.dataset_path}
    dataset = load_dataset("csv", data_files=data_files)

    def preprocess_function(examples: dict) -> dict:
        return tokenizer(examples["text"], truncation=True, padding=True)

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_request.epochs,
        learning_rate=train_request.learning_rate,
        per_device_train_batch_size=train_request.per_device_train_batch_size,
        save_total_limit=1,
        logging_steps=10,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
    )

    trainer.train()
    trainer.save_model(str(output_dir))
