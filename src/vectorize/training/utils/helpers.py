"""Helper functions for training (model, tokenizer, data, training)."""

import random
from pathlib import Path
from uuid import UUID

import numpy as np
import torch
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    concatenate_datasets,
    load_dataset,
)
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession
from torch import stack
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from vectorize.ai_model.repository import get_ai_model_by_id
from vectorize.config.config import settings
from vectorize.datasets.repository import get_dataset_db

from ..exceptions import (
    InvalidDatasetIdError,
    InvalidModelIdError,
    TrainingModelWeightsNotFoundError,
)
from ..triple_dataset import TripletDataset, preprocess_triplet_batch

__all__ = [
    "_set_seed",
    "find_hf_model_dir_svc",
    "load_and_tokenize_datasets",
    "load_model_and_tokenizer",
    "prepare_output_dir",
    "train",
]


def _set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_output_dir(model_path: str) -> Path:
    """Create and return the output directory for the trained model."""
    base_dir = settings.model_upload_dir / "trained_models"
    base_dir.mkdir(parents=True, exist_ok=True)
    model_dir_name = f"{Path(model_path).name}-finetuned"
    output_dir = base_dir / model_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_model_and_tokenizer(model_path: str) -> tuple:
    """Load a Huggingface model and tokenizer from the given path."""
    model_path_obj = Path(model_path)
    if not model_path_obj.exists() or not model_path_obj.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path_obj}")
    try:
        model_load_path = str(find_hf_model_dir_svc(model_path_obj).resolve())
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"No Huggingface model dir (config.json) found in: {model_path_obj}"
        ) from e
    try:
        model = AutoModel.from_pretrained(model_load_path)
    except OSError as e:
        raise TrainingModelWeightsNotFoundError(model_load_path) from e
    tokenizer = AutoTokenizer.from_pretrained(model_load_path)
    return model, tokenizer


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate function that stacks tensors in a batch."""
    return {k: stack([d[k] for d in batch]) for k in batch[0]}


def load_and_tokenize_datasets(
    dataset_paths: list[str], tokenizer: object, batch_size: int
) -> DataLoader:
    """Load datasets and return a DataLoader for triplet training."""
    paths = [Path(p) for p in dataset_paths]
    datasets_list = []
    for p in paths:
        ds_raw = load_dataset("csv", data_files={"train": str(p)})
        if isinstance(ds_raw, (dict, DatasetDict)):
            ds = ds_raw["train"]
        elif isinstance(ds_raw, IterableDataset):
            raise TypeError(f"IterableDataset not supported for {p}")
        else:
            raise TypeError(f"Unexpected dataset type: {type(ds_raw)} for {p}")
        if isinstance(ds, Dataset):
            datasets_list.append(ds)
        else:
            raise TypeError(f"Expected a Dataset, got {type(ds)} for {p}")
    train_data = concatenate_datasets(datasets_list)
    tokenized = train_data.map(
        lambda ex: preprocess_triplet_batch(tokenizer, ex), batched=True
    )
    dataset = TripletDataset(tokenized)
    return DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn)


def train(
    train_ctx: dict,
    epochs: int,
    output_dir: Path | None = None,
    checkpoint_interval: int = 0,
) -> None:
    """Run the training loop for the model, dataloader, optimizer, and criterion.

    Args:
        train_ctx: Dict with model, dataloader, optimizer, criterion, device.
        epochs: Number of epochs to train.
        output_dir: Optional path to save checkpoints.
        checkpoint_interval: Save model every N epochs (if > 0).
    """
    model = train_ctx["model"]
    dataloader = train_ctx["dataloader"]
    optimizer = train_ctx["optimizer"]
    criterion = train_ctx["criterion"]
    device = train_ctx["device"]
    model.to(device)
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
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
        logger.info(
            f"Epoch {epoch + 1} loss: {epoch_loss / len(dataloader):.4f}"
        )
        if (
            output_dir
            and checkpoint_interval > 0
            and (epoch + 1) % checkpoint_interval == 0
        ):
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}"
            with torch.no_grad():
                model.save_pretrained(str(checkpoint_path))
                logger.info(f"Checkpoint saved: {checkpoint_path}")
        if torch.cuda.is_available():
            logger.info(torch.cuda.memory_summary())


def find_hf_model_dir_svc(base_path: Path) -> Path:
    """Recursively search for a subfolder with config.json (Huggingface format)."""
    if (base_path / "config.json").is_file():
        return base_path
    for subdir in base_path.rglob("*"):
        if (subdir / "config.json").is_file():
            return subdir
    raise FileNotFoundError(
        f"No subfolder with config.json found in {base_path}"
    )


async def get_model_path_by_id(
    db: AsyncSession, model_id: str | UUID
) -> str:
    """Lädt das Modell aus der DB und gibt den lokalen Modellpfad zurück."""
    try:
        if not isinstance(model_id, UUID):
            model_id = UUID(model_id)
    except (ValueError, TypeError) as err:
        raise InvalidModelIdError(str(model_id)) from err
    model = await get_ai_model_by_id(db, model_id)
    return str(settings.model_upload_dir / model.model_tag)


async def get_dataset_paths_by_ids(
    db: AsyncSession, dataset_ids: list[str]
) -> list[str]:
    """Lädt die Datasets aus der DB und gibt die lokalen Dateipfade zurück."""
    paths = []
    for dataset_id in dataset_ids:
        try:
            uuid_val = UUID(dataset_id)
        except (ValueError, TypeError) as err:
            raise InvalidDatasetIdError(dataset_id) from err
        dataset = await get_dataset_db(db, uuid_val)
        paths.append(str(Path("data/datasets") / dataset.file_name))
    return paths
