"""Database connection and session management."""

import os
from typing import Final

from dotenv import load_dotenv
from loguru import logger
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from txt2vec.datasets.classification import Classification
from txt2vec.datasets.models import Dataset

load_dotenv()

__all__ = ["close_db", "engine", "init_db", "session"]

DATABASE_URL = os.getenv("DATABASE_URL")

engine: Final = create_async_engine(DATABASE_URL, echo=True)
session: Final = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    """Create database tables and seed initial data."""
    logger.debug("Creating database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    await _seed_db()


async def clear_db():
    """Drop all tables from the database."""
    logger.debug("Dropping all database tables...")

    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.drop_all)
        logger.debug("All database tables dropped successfully")
    except Exception as e:
        logger.error("Error dropping database tables: {}", str(e))
        raise


async def close_db():
    """Close the database connection after dropping all tables."""
    try:
        await clear_db()
    finally:
        logger.info("Closing database connection...")
        await engine.dispose()


async def _seed_db():
    """Seed the database with initial data."""
    try:
        async with session() as db_session:
            result = await db_session.exec(select(Dataset))
            datasets = list(result)

            if not datasets:
                logger.debug("Seeding database with initial data...")

                test_dataset = Dataset(
                    name="example_dataset",
                    classification=Classification.SENTENCE_DUPLES,
                )

                db_session.add(test_dataset)
                await db_session.flush()
                await db_session.commit()
            else:
                logger.debug(
                    "Database already contains {} datasets, skipping seeding",
                    len(datasets),
                )
    except Exception as e:
        logger.error("Error seeding database: {}", str(e))
        raise
