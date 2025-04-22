"""Database connection and session management."""

from typing import Final

from dotenv import load_dotenv
from loguru import logger
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from txt2vec.config.config import db_logging, db_url

load_dotenv()

__all__ = ["close_db", "engine", "init_db", "session"]

engine: Final = create_async_engine(db_url, echo=db_logging)
session: Final = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    """Create database tables and seed initial data."""
    logger.debug("Creating database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


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
