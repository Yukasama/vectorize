import pytest

from txt2vec.upload.exceptions import ModelAlreadyExistsError
from txt2vec.upload.huggingface_service import load_model_and_cache_only


@pytest.mark.asyncio
async def test_load_model_twice_should_raise_conflict():
    model_id = "distilbert-base-uncased"
    tag = "main"

    # Erster Ladeversuch: sollte klappen
    await load_model_and_cache_only(model_id, tag)

    # Zweiter Ladeversuch: sollte Fehler werfen
    with pytest.raises(ModelAlreadyExistsError):
        await load_model_and_cache_only(model_id, tag)
