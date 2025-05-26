"""Service for model training (Transformers Trainer API)."""

from pathlib import Path

import torch
from datasets import load_dataset
from loguru import logger
from torch.nn import TripletMarginLoss
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from vectorize.config import settings

from .exceptions import TrainingDatasetNotFoundError, TrainingModelNotFoundError, TrainingModelWeightsNotFoundError
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
    """Training logic for local models only (no Huggingface download).

    Loads model and dataset, starts training with Triplet-Loss.
    Saves trained models under data/models/trained_models.
    """
    base_dir = settings.model_upload_dir / "trained_models"
    base_dir.mkdir(parents=True, exist_ok=True)
    model_dir_name = f"{Path(train_request.model_path).name}-finetuned"
    output_dir = base_dir / model_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_path_str in train_request.dataset_paths:
        dataset_path = Path(dataset_path_str)
        if not dataset_path.is_file():
            logger.error(
                "Training failed: Dataset file not found: %s",
                dataset_path_str,
            )
            raise TrainingDatasetNotFoundError(dataset_path_str)

    model_path = Path(train_request.model_path)
    if not model_path.exists() or not model_path.is_dir():
        logger.error(f"Model directory not found or not a directory: {model_path}")
        raise TrainingModelNotFoundError(str(model_path))
    try:
        model_load_path = str(_find_hf_model_dir_svc(model_path).resolve())
    except FileNotFoundError as e:
        logger.error(f"No Huggingface model dir (config.json) found in: {model_path}")
        raise TrainingModelNotFoundError(f"No Huggingface model dir (config.json) found in: {model_path}") from e

    try:
        model = AutoModel.from_pretrained(model_load_path)
    except OSError as e:
        logger.error(f"Model weights not found or invalid in {model_load_path}: {e}")
        raise TrainingModelWeightsNotFoundError(model_load_path) from e
    tokenizer = AutoTokenizer.from_pretrained(model_load_path)

    data_files = {"train": train_request.dataset_paths}
    dataset = load_dataset("csv", data_files=data_files)

    def preprocess_function_svc(examples: dict) -> dict:
        anchor = tokenizer(examples["question"], truncation=True, padding=True)
        positive = tokenizer(examples["positive"], truncation=True, padding=True)
        negative = tokenizer(examples["negative"], truncation=True, padding=True)
        return {
            "anchor_input_ids": anchor["input_ids"],
            "anchor_attention_mask": anchor["attention_mask"],
            "positive_input_ids": positive["input_ids"],
            "positive_attention_mask": positive["attention_mask"],
            "negative_input_ids": negative["input_ids"],
            "negative_attention_mask": negative["attention_mask"],
        }

    tokenized = dataset.map(preprocess_function_svc, batched=True)
    train_data = tokenized["train"]

    class TripletDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data["anchor_input_ids"])
        def __getitem__(self, idx):
            return {
                "anchor_input_ids": torch.tensor(self.data["anchor_input_ids"][idx]),
                "anchor_attention_mask": torch.tensor(self.data["anchor_attention_mask"][idx]),
                "positive_input_ids": torch.tensor(self.data["positive_input_ids"][idx]),
                "positive_attention_mask": torch.tensor(self.data["positive_attention_mask"][idx]),
                "negative_input_ids": torch.tensor(self.data["negative_input_ids"][idx]),
                "negative_attention_mask": torch.tensor(self.data["negative_attention_mask"][idx]),
            }

    train_dataset = TripletDataset(train_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_request.per_device_train_batch_size,
        shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_request.learning_rate)
    criterion = TripletMarginLoss(margin=1.0, p=2)

    for epoch in range(train_request.epochs):
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            anchor = model(
                input_ids=batch["anchor_input_ids"].to(device),
                attention_mask=batch["anchor_attention_mask"].to(device),
            )[0][:, 0, :]
            positive = model(
                input_ids=batch["positive_input_ids"].to(device),
                attention_mask=batch["positive_attention_mask"].to(device),
            )[0][:, 0, :]
            negative = model(
                input_ids=batch["negative_input_ids"].to(device),
                attention_mask=batch["negative_attention_mask"].to(device),
            )[0][:, 0, :]
            loss = criterion(anchor, positive, negative)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        logger.info(f"Epoch {epoch+1} loss: {epoch_loss/len(train_loader):.4f}")

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
