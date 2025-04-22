"""Seed the database with initial data."""

from loguru import logger
from sqlmodel import select

from txt2vec.config.db import clear_db, session
from txt2vec.datasets.classification import Classification
from txt2vec.datasets.models import Dataset

__all__ = ["seed_db"]


async def seed_db():
    """Seed the database with initial data."""
    try:
        async with session() as db_session:
            result = await db_session.exec(select(Dataset))
            datasets = list(result)

            if datasets:
                await clear_db()

            logger.debug("Seeding database with initial data...")

            test_dataset = Dataset(
                name="example_dataset",
                file_name="example_dataset.csv",
                classification=Classification.SENTENCE_DUPLES,
                rows=5,
            )

            db_session.add(test_dataset)
            await db_session.flush()
            await db_session.commit()

    except Exception as e:
        logger.error(str(e))
        raise
