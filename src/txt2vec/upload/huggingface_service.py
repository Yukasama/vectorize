"""Service zum Laden und Speichern von Hugging Face Modellen.

Dieses Modul lädt Modelle von Hugging Face, cached sie lokal und speichert sie
in der Datenbank, falls sie noch nicht vorhanden sind.
"""

import os

from huggingface_hub import snapshot_download
from huggingface_hub.utils import EntryNotFoundError
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from txt2vec.upload.exceptions import (
    InvalidModelError,
    NoValidModelsFoundError,
)

_models = {}


async def load_model_and_cache_only(model_id: str, tag: str | None = None) -> None:
    """Loads a Hugging Face model and caches it locally with validation."""
    tag = tag or "main"
    key = f"{model_id}@{tag}"
    if key in _models:
        logger.info(f"Model '{key}' is already in Cache.")
        return

    try:
        snapshot_path = snapshot_download(
            repo_id=model_id,
            revision=tag,
            cache_dir="./hf_cache",
            allow_patterns=["*.safetensors", "*.json"],  # ✅ Punkt 5
        )

        safetensors_files = [
            f for f in os.listdir(snapshot_path) if f.endswith(".safetensors")
        ]

        if len(safetensors_files) != 1:
            raise NoValidModelsFoundError  # ✅ Punkt 4

        tokenizer = AutoTokenizer.from_pretrained(snapshot_path)
        model = AutoModelForSequenceClassification.from_pretrained(snapshot_path)

        _models[key] = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        logger.info(f"Model '{key}' successfully loaded and cached.")
    except EntryNotFoundError:
        raise FileNotFoundError(f"Model '{model_id}' with tag '{tag}' not found.")  # ✅ Punkt 1
    except Exception as e:
        logger.exception(f"Error loading the model '{key}'")
        raise InvalidModelError from e
