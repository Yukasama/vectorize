"""Utility functions for handling model tags."""

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from ..models import AIModel

__all__ = ["next_available_tag"]


async def next_available_tag(
    db: AsyncSession,
    base_tag: str,
) -> str:
    """Return unique tag based on base_tag.

    Automatically ensures model_tag is unique by appending -N if needed.

    Args:
        db: AsyncSession
            Database session instance.
        base_tag: str
            The base tag to derive a unique tag from.

    Returns:
        str: A unique tag derived from base_tag.
    """
    stmt = select(AIModel.model_tag).where(
        (AIModel.model_tag == base_tag) | (AIModel.model_tag.like(f"{base_tag}-%"))
    )
    rows = await db.exec(stmt)
    existing = {row[0] for row in rows.all()}

    if base_tag not in existing:
        return base_tag

    counters = [
        int(tag.split("-")[-1])
        for tag in existing
        if tag.startswith(f"{base_tag}-") and tag.split("-")[-1].isdigit()
    ]
    next_counter = (max(counters) if counters else 1) + 1
    return f"{base_tag}-{next_counter}"
