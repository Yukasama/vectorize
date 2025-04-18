from loguru import logger

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from huggingface_hub import snapshot_download

# Globale Pipeline (wird beim Laden gesetzt)
CLASSIFIER = None


def load_model_with_tag(model_id: str, tag: str):

    global CLASSIFIER

    # Lade lokalen Snapshot vom HF-Model mit Tag
    snapshot_path = snapshot_download(
        repo_id=model_id, revision=tag, cache_dir="./hf_cache"
    )
    logger.debug("Model snapshot path: {}", snapshot_path)

    # Lade Tokenizer und Model
    tokenizer = AutoTokenizer.from_pretrained(snapshot_path)
    model = AutoModelForSequenceClassification.from_pretrained(snapshot_path)

    # Erstelle Pipeline
    CLASSIFIER = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


def get_classifier():
    """
    Gibt die geladene Pipeline zurück.

    :raises ValueError: Wenn kein Modell geladen wurde.
    :return: Die geladene Pipeline.
    """
    global CLASSIFIER
    if CLASSIFIER is None:
        raise ValueError("Kein Modell geladen.")
    return CLASSIFIER


def reset_classifier():
    """
    Setzt die globale Pipeline zurück (nur für Tests oder Debugging).
    """
    global CLASSIFIER
    CLASSIFIER = None
