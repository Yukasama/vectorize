"""Actions repository."""

from datetime import UTC, datetime, timedelta
from typing import Any

from loguru import logger
from sqlalchemy import ColumnElement, String, cast, literal
from sqlmodel import func, or_, select
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.common.task_status import TaskStatus
from vectorize.dataset.task_model import UploadDatasetTask
from vectorize.synthesis.models import SynthesisTask
from vectorize.upload.models import UploadTask

from .actions_model import ModelType


def active(model: ModelType) -> ColumnElement[bool]:
    one_hour_ago = datetime.now(tz=UTC) - timedelta(hours=1)
    return or_(
        model.task_status == TaskStatus.PENDING,
        model.task_status == TaskStatus.QUEUED,
        func.coalesce(model.end_date, func.now()) >= one_hour_ago,
    )


def base_query(model: ModelType, tag: str):
    return select(
        model.id,
        model.task_status,
        model.created_at,
        model.end_date,
        cast(literal(tag), String).label("task_type"),
    ).where(active(model))


async def get_actions_db(db: AsyncSession) -> list[dict[str, Any]]:
    upload_q = base_query(UploadTask, "model_upload")
    synthesis_q = base_query(SynthesisTask, "synthesis")
    dataset_q = base_query(UploadDatasetTask, "dataset_upload")

    union_subq = upload_q.union_all(synthesis_q, dataset_q).subquery()
    stmt = select(
        union_subq.c.id,
        union_subq.c.task_status,
        union_subq.c.created_at,
        union_subq.c.end_date,
        union_subq.c.task_type,
    )

    result = await db.exec(stmt)
    logger.debug(actions_length=len(result.all()))
    return result.all()
