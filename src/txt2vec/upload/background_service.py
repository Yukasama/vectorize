"""Background service to handle GitHub and Huggingface upload processes."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from txt2vec.config.db import get_session
from txt2vec.upload.models import UploadTask


async def write_to_database(
    db: Annotated[AsyncSession, Depends(get_session)], upload_task: UploadTask
) -> dict:
    """FuBar."""
    db.add(upload_task)
    await db.commit()
    await db.refresh(upload_task)

    return {}
