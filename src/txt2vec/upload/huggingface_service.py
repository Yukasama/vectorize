"""Service zum lokalen Laden und Cachen von HF-Modellen."""

from huggingface_hub import snapshot_download
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from txt2vec.upload.exceptions import InvalidModelError

__all__ = ["get_classifier", "load_model_HF", "reset_models"]

_models = {}


def load_model_HF(model_id: str, tag: str) -> None:
    """Lädt Modell von HF und cached es."""
    key = f"{model_id}@{tag}"
    if key in _models:
        logger.info(f"'{key}' bereits geladen.")
        return

    try:
        logger.info(f"Lade '{key}'...")
        snapshot_path = snapshot_download(
            repo_id=model_id, revision=tag, cache_dir="./hf_cache"
        )
        tokenizer = AutoTokenizer.from_pretrained(snapshot_path)
        model = AutoModelForSequenceClassification.from_pretrained(snapshot_path)
        _models[key] = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        logger.info(f"'{key}' geladen.")
    except Exception as e:
        logger.exception(f"Fehler bei '{key}'")
        raise InvalidModelError() from e


def get_classifier(model_id: str, tag: str):
    """Gibt Pipeline zu einem geladenen Modell zurück."""
    key = f"{model_id}@{tag}"
    if key not in _models:
        raise ValueError(f"'{key}' nicht geladen.")
    return _models[key]


def reset_models():
    """Leert den Modell-Cache."""
    logger.info("Modell-Cache geleert.")
    _models.clear()
