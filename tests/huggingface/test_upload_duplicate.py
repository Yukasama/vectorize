"""Test für das Laden eines bereits vorhandenen Huggingface-Modells."""
import pytest

from txt2vec.upload.exceptions import ModelAlreadyExistsError
from txt2vec.upload.huggingface_service import load_model_and_cache_only


@pytest.mark.asyncio
async def test_load_distilbert_model() -> None:
    """Testet das Laden eines bereits vorhandenen distilbert-base-uncased Modells."""
    model_id = "distilbert-base-uncased"
    tag = "main"

    try:
        await load_model_and_cache_only(model_id, tag)
    except ModelAlreadyExistsError:
        pass
    except Exception as e:
        pytest.fail(f"Model loading failed with error: {e}")
