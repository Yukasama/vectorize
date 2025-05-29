"""Hilfsfunktionen für das Training (Modell, Tokenizer, Daten, Training)."""

from pathlib import Path

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    concatenate_datasets,
    load_dataset,
)
from loguru import logger
from torch.utils.data import DataLoader, default_collate
from transformers import AutoModel, AutoTokenizer

from vectorize.config import settings

from ..datasets import TripletDataset, preprocess_triplet_batch
from ..exceptions import TrainingModelNotFoundError, TrainingModelWeightsNotFoundError

__all__ = [
    "find_hf_model_dir_svc",
    "load_and_tokenize_datasets",
    "load_model_and_tokenizer",
    "prepare_output_dir",
    "train",
]


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
        raise TrainingModelNotFoundError(str(model_path_obj))
    try:
        model_load_path = str(find_hf_model_dir_svc(model_path_obj).resolve())
    except FileNotFoundError as e:
        raise TrainingModelNotFoundError(
            f"No Huggingface model dir (config.json) found in: {model_path_obj}"
        ) from e
    try:
        model = AutoModel.from_pretrained(model_load_path)
    except OSError as e:
        raise TrainingModelWeightsNotFoundError(model_load_path) from e
    tokenizer = AutoTokenizer.from_pretrained(model_load_path)
    return model, tokenizer


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
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=default_collate
    )


def train(
    train_ctx: dict,
    epochs: int,
) -> None:
    """Run the training loop for the model, dataloader, optimizer, and criterion."""
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
        logger.info(f"Epoch {epoch + 1} loss: {epoch_loss / len(dataloader):.4f}")


def find_hf_model_dir_svc(base_path: Path) -> Path:
    """Recursively search for a subfolder with config.json (Huggingface format)."""
    if (base_path / "config.json").is_file():
        return base_path
    for subdir in base_path.rglob(""):
        if (subdir / "config.json").is_file():
            return subdir
    raise FileNotFoundError(f"No subfolder with config.json found in {base_path}")
