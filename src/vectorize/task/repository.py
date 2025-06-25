"""Tasks repository."""

from collections.abc import Sequence

from loguru import logger
from sqlmodel import text
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.dataset.task_model import UploadDatasetTask
from vectorize.evaluation.models import EvaluationTask
from vectorize.synthesis.models import SynthesisTask
from vectorize.training.models import TrainingTask
from vectorize.upload.models import UploadTask

from .query_builder import build_query
from .schemas import TaskFilters

__all__ = ["get_tasks_db"]


async def get_tasks_db(db: AsyncSession, params: TaskFilters) -> Sequence:
    """Retrieve tasks from database with filtering and pagination.

    Aggregates tasks from multiple types (upload, synthesis, dataset) with
    comprehensive filtering and pagination support.

    Args:
        db: Database session for executing queries.
        params: Filter parameters containing limit, offset, completed status,
                specific statuses, and time window criteria.

    Returns:
        Sequence of task action rows ordered by creation date (newest first).
        Each row contains id, task_status, created_at, end_date, and task_type.
    """
    status_set = set(params.statuses or [])

    upload_q = build_query(
        UploadTask,
        "model_upload",
        completed=params.completed,
        statuses=status_set,
        hours=params.within_hours,
    )
    synth_q = build_query(
        SynthesisTask,
        "synthesis",
        completed=params.completed,
        statuses=status_set,
        hours=params.within_hours,
    )
    dataset_q = build_query(
        UploadDatasetTask,
        "dataset_upload",
        completed=params.completed,
        statuses=status_set,
        hours=params.within_hours,
    )
    training_q = build_query(
        TrainingTask,
        "training",
        completed=params.completed,
        statuses=status_set,
        hours=params.within_hours,
    )
    evaluation_q = build_query(
        EvaluationTask,
        "evaluation",
        completed=params.completed,
        statuses=status_set,
        hours=params.within_hours,
    )

    stmt = (
        upload_q.union_all(synth_q, dataset_q, training_q, evaluation_q)
        .order_by(text("created_at DESC"))
        .limit(params.limit)
        .offset(params.offset or 0)
    )
    result = await db.exec(stmt)

    rows = result.all()
    logger.debug("Tasks fetched from DB", count=len(rows), params=str(params))
    return rows
