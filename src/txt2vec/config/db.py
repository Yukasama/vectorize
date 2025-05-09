"""Database connection and session management."""

from collections.abc import AsyncGenerator
from typing import Final

from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from .config import settings

__all__ = ["engine", "get_session"]

engine: Final = create_async_engine(
    settings.db_url,
    connect_args={"check_same_thread": False, "timeout": settings.db_timeout},
    echo=settings.db_logging,
    future=settings.db_future,
    pool_timeout=settings.db_pool_timeout,
    pool_size=settings.db_pool_size,
    pool_pre_ping=settings.db_pool_pre_ping,
    max_overflow=settings.db_max_overflow,
    pool_recycle=settings.db_pool_recycle,
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
