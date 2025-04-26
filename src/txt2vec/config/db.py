"""Database connection and session management."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import StaticPool
from sqlmodel.ext.asyncio.session import AsyncSession

from txt2vec.config.config import db_logging, db_url

__all__ = ["engine", "get_session"]

engine = create_async_engine(
    db_url,
    poolclass=StaticPool if db_url.endswith(":memory:") else None,
    connect_args={"check_same_thread": False} if db_url.endswith(":memory:") else {},
    echo=db_logging,
    future=True,
)


async def get_session() -> AsyncGenerator[AsyncSession]:
    """Get session for database operations."""
    async with AsyncSession(engine, expire_on_commit=False) as session:
        yield session
