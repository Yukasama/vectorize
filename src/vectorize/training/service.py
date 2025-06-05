"""Service layer for training (now SBERT/SentenceTransformer only)."""

# No longer needed: all training logic is in tasks.py using sentence-transformers.
# This file can be left empty or used for future service abstractions.

import gc
import os
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import Dataset, DataLoader

from .utils.safetensors_finder import find_safetensors_file


class InputExampleDataset(Dataset):
    """Kapselt eine Liste von InputExamples für den DataLoader."""
    def __init__(self, examples):
        self.examples = examples

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)


def train_sbert_triplet_service(
    model_path: str,
    train_request: Any,  # Pydantic model, aber auch dict für Tests möglich
    dataset_paths: list[str],
    output_dir: str,
    progress_callback=None,
) -> None:
    """Trainiert ein SentenceTransformer (SBERT) Modell mit CosineSimilarityLoss auf Tripletdaten (Contrastive Learning)."""
    # Use safetensors finder for model loading
    safetensors_path = find_safetensors_file(model_path)
    if safetensors_path:
        model_dir = os.path.dirname(safetensors_path)
        logger.debug(f"Found .safetensors file for model: {safetensors_path} (using dir: {model_dir})")
        model = SentenceTransformer(model_dir)
    else:
        logger.debug("No .safetensors file found, loading model from original path.")
        model = SentenceTransformer(model_path)
    # Padding-Token setzen, falls nicht vorhanden
    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Triplet-Daten aus JSONL(s) laden und für Contrastive-Loss aufbereiten
    def extract_triplet(row):
        if all(k in row for k in ("Question", "Positive", "Negative")):
            return [row["Question"], row["Positive"], row["Negative"]]
        else:
            raise ValueError(
                "Jede Zeile muss die Keys 'Question', 'Positive', 'Negative' enthalten. Gefunden: %s" % list(row.keys())
            )

    df = pd.read_json(dataset_paths[0], lines=True)
    train_examples = []
    for _, row in df.iterrows():
        q, pos, neg = extract_triplet(row)
        train_examples.append(InputExample(texts=[q, pos], label=1.0))
        train_examples.append(InputExample(texts=[q, neg], label=-1.0))

    val_examples = None
    if len(dataset_paths) > 1:
        val_df = pd.read_json(dataset_paths[1], lines=True)
        val_examples = []
        for _, row in val_df.iterrows():
            q, pos, neg = extract_triplet(row)
            val_examples.append(InputExample(texts=[q, pos], label=1.0))
            val_examples.append(InputExample(texts=[q, neg], label=-1.0))
    else:
        val_split = int(0.1 * len(train_examples))
        val_examples = train_examples[:val_split]
        train_examples = train_examples[val_split:]

    train_dataset = InputExampleDataset(train_examples)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_request.per_device_train_batch_size,
        shuffle=True,
        num_workers=train_request.dataloader_num_workers or 0,
    )
    loss = losses.CosineSimilarityLoss(model)

    fit_kwargs = {
        "epochs": train_request.epochs,
        "warmup_steps": train_request.warmup_steps or 0,
        "show_progress_bar": train_request.show_progress_bar if train_request.show_progress_bar is not None else True,
        "output_path": output_dir,
    }
    for key in [
        "optimizer_name", "scheduler", "weight_decay", "max_grad_norm", "use_amp",
        "evaluation_steps", "save_best_model", "save_each_epoch", "save_optimizer_state", "device"
    ]:
        value = getattr(train_request, key, None)
        if value is not None:
            fit_kwargs[key] = value

    logger.info(
        "Starte SBERT Contrastive-Training mit Parametern: %s | Modell-Ordner: %s | Output: %s",
        fit_kwargs,
        model_dir if safetensors_path else model_path,
        output_dir,
    )
    model.fit(
        train_objectives=[(train_dataloader, loss)],
        **fit_kwargs,
    )
    model.save(output_dir)
    logger.info(f"SBERT Modell gespeichert unter {output_dir} (trainiert von: {model_dir if safetensors_path else model_path})")
    # Nach dem Speichern: Modell als AIModel in die DB eintragen
    try:
        from vectorize.ai_model.models import AIModel
        from vectorize.ai_model.model_source import ModelSource
        from vectorize.ai_model.repository import save_ai_model_db
        import datetime
        from sqlmodel import Session, create_engine
        # Parent-Tag bestimmen
        parent_tag = os.path.basename(model_path.rstrip("/"))
        tag_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
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
        logger.info(f"Finetuned Modell in DB gespeichert: {new_model_tag}")
    except Exception as exc:
        logger.error(f"Fehler beim Speichern des finetuned Modells in die DB: {exc}")
    try:
        del model
    except Exception as exc:
        logger.warning("Cleanup fehlgeschlagen (model): %s", exc)
    gc.collect()
