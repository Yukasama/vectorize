"""Service zum Laden und Speichern von Hugging Face Modellen.

Dieses Modul lädt Modelle von Hugging Face, cached sie lokal und speichert sie
in der Datenbank, falls sie noch nicht vorhanden sind.
"""

from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import EntryNotFoundError
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from txt2vec.upload.exceptions import (
    InvalidModelError,
    ModelAlreadyExistsError,
    NoValidModelsFoundError,
)

_models = {}


async def load_model_and_cache_only(model_id: str, tag: str) -> None:  # noqa: RUF029
    """Load a Hugging Face model and cache it locally if not already cached.

    Downloads the model and tokenizer from Hugging Face using the given
    model_id and tag, checks for valid safetensors, and stores the pipeline
    in a local cache. Raises if the model is not found or invalid.

    Args:
        model_id: The Hugging Face model repository ID.
        tag: The revision or tag to download.
    """
    key = f"{model_id}@{tag}"

    if key in _models:
        logger.info("Model is already in Cache.", modelKey=key)
        raise ModelAlreadyExistsError(key)  # Hier wird der Fehler geworfen!

    try:
        snapshot_path = snapshot_download(
            repo_id=model_id,
            revision=tag,
            cache_dir="./hf_cache",
            allow_patterns=["*.safetensors", "*.json"],
        )

        safetensors_files = [
            f.name for f in Path(snapshot_path).iterdir()
            if f.name.endswith(".safetensors")
        ]

        if len(safetensors_files) != 1:
            raise NoValidModelsFoundError

        tokenizer = AutoTokenizer.from_pretrained(snapshot_path)
        model = AutoModelForSequenceClassification.from_pretrained(snapshot_path)

        _models[key] = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        logger.info("Model successfully loaded and cached.", modelKey=key)

    except EntryNotFoundError as e:
        logger.debug(
            "Model not found on Hugging Face.",
            modelId=model_id,
            tag=tag,
            error=str(e),
        )
        raise FileNotFoundError(
            "Model not found on Hugging Face.",
            {"modelId": model_id, "tag": tag, "error": str(e)},
        ) from e

    except Exception as e:
        logger.exception("Error loading model.", modelKey=key)
        raise InvalidModelError from e
