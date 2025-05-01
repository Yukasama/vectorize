import pytest
from sqlmodel import SQLModel

from txt2vec.ai_model.models import AIModel
from txt2vec.ai_model.utils.tag_helpers import next_available_tag
from txt2vec.config.db import get_session


@pytest.mark.asyncio
async def test_next_available_tag_increments_correctly():
    session_gen = get_session()
    session = await anext(session_gen)
    try:
        await session.exec("DELETE FROM aimodel")

        base_tag = "distilbert-main"

        model1 = AIModel(model_tag=base_tag, name="Model1")
        model2 = AIModel(model_tag=f"{base_tag}-1", name="Model2")
        model3 = AIModel(model_tag=f"{base_tag}-2", name="Model3")

        session.add_all([model1, model2, model3])
        await session.commit()

        result = await next_available_tag(session, base_tag)
        assert result == f"{base_tag}-3"
    finally:
        await session_gen.aclose()
