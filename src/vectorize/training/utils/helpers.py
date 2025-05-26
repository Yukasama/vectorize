"""Hilfsfunktionen für das Training (Modell, Tokenizer, Daten, Training)."""

from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from loguru import logger
from torch.utils.data import DataLoader, default_collate
from transformers import AutoModel, AutoTokenizer

from vectorize.config import settings

from ..datasets import TripletDataset, preprocess_triplet_batch
from ..exceptions import TrainingModelNotFoundError, TrainingModelWeightsNotFoundError

__all__ = [
    "_find_hf_model_dir_svc",
    "_load_and_tokenize_datasets",
    "_load_model_and_tokenizer",
    "_prepare_output_dir",
    "_train",
]


def _prepare_output_dir(model_path: str) -> Path:
    base_dir = settings.model_upload_dir / "trained_models"
    base_dir.mkdir(parents=True, exist_ok=True)
    model_dir_name = f"{Path(model_path).name}-finetuned"
    output_dir = base_dir / model_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _load_model_and_tokenizer(model_path: str) -> tuple:
    model_path = Path(model_path)
    if not model_path.exists() or not model_path.is_dir():
        raise TrainingModelNotFoundError(str(model_path))
    try:
        model_load_path = str(_find_hf_model_dir_svc(model_path).resolve())
    except FileNotFoundError as e:
        raise TrainingModelNotFoundError(
            f"No Huggingface model dir (config.json) found in: {model_path}"
        ) from e
    try:
        model = AutoModel.from_pretrained(model_load_path)
    except OSError as e:
        raise TrainingModelWeightsNotFoundError(model_load_path) from e
    tokenizer = AutoTokenizer.from_pretrained(model_load_path)
    return model, tokenizer


def _load_and_tokenize_datasets(
    dataset_paths: list[str], tokenizer: object, batch_size: int
) -> DataLoader:
    paths = [Path(p) for p in dataset_paths]
    datasets_list = [
        load_dataset("csv", data_files={"train": str(p)})["train"] for p in paths
    ]
    train_data = concatenate_datasets(datasets_list)
    tokenized = train_data.map(
        lambda ex: preprocess_triplet_batch(tokenizer, ex), batched=True
    )
    dataset = TripletDataset(tokenized)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=default_collate
    )


def _train(
    train_ctx: dict,
    epochs: int,
) -> None:
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


def _find_hf_model_dir_svc(base_path: Path) -> Path:
    """Suche rekursiv nach einem Unterordner mit config.json (Huggingface-Format)."""
    if (base_path / "config.json").is_file():
        return base_path
    for subdir in base_path.rglob(""):
        if (subdir / "config.json").is_file():
            return subdir
    raise FileNotFoundError(f"Kein Unterordner mit config.json in {base_path}")
