"""Test für das Laden eines ungültigen Huggingface-Modells."""
import pytest

from txt2vec.upload.exceptions import InvalidModelError
from txt2vec.upload.huggingface_service import load_model_and_cache_only


@pytest.mark.asyncio
async def test_load_invalid_model_should_fail() -> None:
    """Testet, dass das Laden eines ungültigen Modells fehlschlägt."""
    # Ungültige Model-ID, die es garantiert nicht gibt
    model_id = "nonexistent-model-id-xyz1234567890"
    tag = "main"

    # Erwartet, dass ein Fehler geworfen wird
    with pytest.raises(InvalidModelError):
        await load_model_and_cache_only(model_id, tag)
