"""Seed the database with initial data."""

from sqlmodel.ext.asyncio.session import AsyncSession

from txt2vec.datasets.models import Classification, Dataset


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
        )
    )
    await session.commit()
