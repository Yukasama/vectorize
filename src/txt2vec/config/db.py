"""Database connection and session management."""

from collections.abc import AsyncGenerator
from typing import Final

from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import StaticPool
from sqlmodel.ext.asyncio.session import AsyncSession

from txt2vec.config import settings

__all__ = ["engine", "get_session"]

engine: Final = create_async_engine(
    settings.db_url,
    poolclass=StaticPool if settings.db_url.endswith(":memory:") else None,
    connect_args={"check_same_thread": False}
    if settings.db_url.endswith(":memory:")
    else {},
    echo=settings.db_logging,
    future=True,
)


async def get_session() -> AsyncGenerator[AsyncSession]:
    """Get session for database operations."""
    async with AsyncSession(engine, expire_on_commit=False) as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
