"""Service layer for training (now SBERT/SentenceTransformer only)."""

import gc
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from loguru import logger
from sentence_transformers import InputExample, SentenceTransformer, losses
from sqlmodel import Session, create_engine
from torch.utils.data import DataLoader, Dataset

from vectorize.ai_model.model_source import ModelSource
from vectorize.ai_model.models import AIModel

from .schemas import TrainRequest
from .utils.safetensors_finder import find_safetensors_file


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
            "Found .safetensors file for model: %s (using dir: %s)",
            safetensors_path,
            model_dir,
        )
        return SentenceTransformer(str(model_dir))
    logger.debug("No .safetensors file found, loading model from original path.")
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
    examples = []
    for _, row in df.iterrows():
        q, pos, neg = _extract_triplet(row)
        examples.extend([
            InputExample(texts=[q, pos], label=1.0),
            InputExample(texts=[q, neg], label=-1.0),
        ])
    return examples


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
        logger.info("Finetuned model saved in DB: %s", new_model_tag)
    except Exception as exc:
        logger.error("Error saving finetuned model to DB: %s", exc)


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
        "Starting SBERT contrastive training with parameters: %s | Model dir: %s | "
        "Output: %s",
        fit_kwargs,
        model_path,
        output_dir,
    )
    model.fit(
        train_objectives=[(train_dataloader, loss)],
        **fit_kwargs,
    )
    model.save(output_dir)
    logger.info(
        "SBERT model saved at %s (trained from: %s)",
        output_dir,
        model_path,
    )
    _save_finetuned_model(model_path)
    try:
        del model
    except Exception as exc:
        logger.warning("Cleanup failed (model): %s", exc)
    gc.collect()
