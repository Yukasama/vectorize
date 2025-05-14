import pytest

from txt2vec.upload.huggingface_service import load_model_and_cache_only


@pytest.mark.asyncio
async def test_load_distilbert_model():
    model_id = "distilbert-base-uncased"
    tag = "main"

    try:
        await load_model_and_cache_only(model_id, tag)
    except Exception as e:
        pytest.fail(f"Model loading failed with error: {e}")
