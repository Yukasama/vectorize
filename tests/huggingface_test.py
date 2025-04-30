"""Tests für das Modell-Service-Modul.

Testfunktionen:
- test_load_and_get_model: Testet das Laden eines gültigen Modells und die
  Pipeline-Ausgabe.
- test_invalid_model_raises_error: Testet, ob beim Laden eines ungültigen Modells
  eine InvalidModelError ausgelöst wird.
"""

import pytest
from transformers import Pipeline

from txt2vec.upload.huggingface_service import reset_models, load_model_HF, get_classifier
from txt2vec.upload.exceptions import InvalidModelError


def test_load_and_get_model():
    """Testet das Laden eines gültigen Modells und die Verarbeitung eines Textes."""
    reset_models()

    model_id = "distilbert-base-uncased"
    tag = "main"

    load_model_HF(model_id, tag)
    classifier = get_classifier(model_id, tag)

    assert isinstance(classifier, Pipeline)
    result = classifier("I love this!")

    assert isinstance(result, list)
    assert "label" in result[0]
    assert "score" in result[0]


def test_invalid_model_raises_error():
    """Testet, ob eine InvalidModelError ausgelöst wird.

    Versucht, ein nicht existierendes Modell zu laden, und überprüft, ob die
    entsprechende Ausnahme ausgelöst wird.
    """
    reset_models()

    with pytest.raises(InvalidModelError):
        load_model_HF("nonexistent-model-xyz", "no-tag")
