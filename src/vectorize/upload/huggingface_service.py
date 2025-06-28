"""Service for loading and saving Hugging Face models.

This module loads models from Hugging Face, caches them locally, and stores them
in the database if they are not already present.
"""

import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import EntryNotFoundError
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.pipelines import pipeline

from .exceptions import (
    InvalidModelError,
    NoValidModelsFoundError,
)

_models = {}


__all__ = ["load_huggingface_model_and_cache_only_svc",
    "remove_huggingface_model_from_memory_svc"]


async def load_huggingface_model_and_cache_only_svc(  # noqa: RUF029 NOSONAR
    model_tag: str, revision: str
) -> None:
    """Load a Hugging Face model and cache it locally if not already cached.

    Downloads the model and tokenizer from Hugging Face using the given
    model_tag and revision, checks for valid safetensors, and stores the pipeline
    in a local cache. Raises if the model is not found or invalid.

    Args:
        model_tag: The Hugging Face model repository tag.
        revision: The revision or version to download.

    Raises:
        NoValidModelsFoundError: If no valid .safetensors file is found or
            more than one exists.
        FileNotFoundError: If the model is not found on Hugging Face.
        InvalidModelError: If an error occurs while loading the model or
            tokenizer.
    """
    key = f"{model_tag}@{revision}"

    if key in _models:
        logger.info("Model is already in Cache.", modelKey=key)
        return

    try:
        snapshot_path = snapshot_download(
            repo_id=model_tag,
            revision=revision,
            cache_dir="data/models",
            allow_patterns=["*.safetensors", "*.json"],
        )

        safetensors_files = [
            f.name
            for f in Path(snapshot_path).iterdir()
            if f.name.endswith(".safetensors")
        ]

        if len(safetensors_files) != 1:
            raise NoValidModelsFoundError

        tokenizer = AutoTokenizer.from_pretrained(snapshot_path)
        model = AutoModelForSequenceClassification.from_pretrained(snapshot_path)

        _models[key] = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)  # type: ignore
        logger.info("Model successfully loaded and cached.", modelKey=key)

    except EntryNotFoundError as e:
        logger.debug(
            "Model not found on Hugging Face.",
            modelTag=model_tag,
            revision=revision,
            error=str(e),
        )
        raise FileNotFoundError(
            "Model not found on Hugging Face.",
            {"modelTag": model_tag, "revision": revision, "error": str(e)},
        ) from e

    except Exception as e:
        logger.exception("Error loading model.", modelKey=key)
        raise InvalidModelError from e


async def remove_huggingface_model_from_memory_svc(model_tag: str) -> None:  # noqa: RUF029 NOSONAR
    """Remove a Hugging Face model from the disk.

    Args:
        model_tag (str): The tag of the Hugging Face model repository.
    """
    base_path = Path("/app/data/models")
    model_folder = base_path / model_tag

    if model_folder.exists() and model_folder.is_dir():
        shutil.rmtree(model_folder)
        logger.warning(f"Deleted model folder from disk: {model_folder}")
    else:
        logger.warning(f"Model folder not found on disk: {model_folder}")
