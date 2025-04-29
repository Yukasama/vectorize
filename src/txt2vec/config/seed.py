"""Seed the database with initial data."""

from sqlmodel.ext.asyncio.session import AsyncSession

from txt2vec.ai_model.model_source import ModelSource
from txt2vec.ai_model.models import AIModel
from txt2vec.datasets.classification import Classification
from txt2vec.datasets.models import Dataset


async def seed_db(session: AsyncSession) -> None:
    """Seed the database with initial data.

    Args:
        session: The SQLModel async database session.
    """
    session.add(
        Dataset(
            name="example_dataset",
            file_name="example_dataset.csv",
            classification=Classification.SENTENCE_DUPLES,
            rows=5,
        ),
    )
    session.add(
        AIModel(
            name="example_model",
            source=ModelSource.LOCAL,
            model_tag="pytorch_model",
        ),
    )
    await session.commit()
