"""Helpers for tasks repository."""

from datetime import UTC, datetime, timedelta

from sqlalchemy import ColumnElement, String, cast, func, literal, or_, select
from sqlmodel import true

from vectorize.ai_model.models import AIModel
from vectorize.common.task_status import TaskStatus

__all__ = ["build_query"]


_COMPLETED = {TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELED}
_UNFINISHED = {TaskStatus.QUEUED, TaskStatus.PENDING}


def build_query(  # noqa: ANN201
    model,  # noqa: ANN001
    tag: str,
    *,
    completed: bool | None,
    statuses: set[TaskStatus],
    hours: int,
):
    """Build base SQL query for a task model with filters applied.

    Args:
        model: SQLModel table class to query.
        tag: String identifier for task type.
        completed: Task completion filter.
        statuses: Set of task statuses to include.
        hours: Time window for filtering.

    Returns:
        SQLAlchemy Select query with standardized columns and filters.
    """
    if hasattr(model, "trained_model_id"):
        model_table = model.__table__
        ai_table = AIModel.__table__  # type: ignore

        join_expr = model_table.outerjoin(
            ai_table, model_table.c.trained_model_id == ai_table.c.id
        )

        return (
            select(
                model_table.c.id,
                ai_table.c.model_tag.label("tag"),
                model_table.c.task_status,
                model_table.c.created_at,
                model_table.c.end_date,
                model_table.c.error_msg,
                cast(literal(tag), String).label("task_type"),
            )
            .select_from(join_expr)
            .where(_status_filter(model, completed=completed, statuses=statuses))
            .where(_time_filter(model, hours=hours))
        )

    if hasattr(model, "evaluation_metrics"):
        model_table = model.__table__
        ai_table = AIModel.__table__  # type: ignore

        join_expr = model_table.outerjoin(
            ai_table, model_table.c.model_id == ai_table.c.id
        )

        return (
            select(
                model_table.c.id,
                ai_table.c.model_tag.label("tag"),
                model_table.c.task_status,
                model_table.c.created_at,
                model_table.c.end_date,
                model_table.c.error_msg,
                cast(literal(tag), String).label("task_type"),
            )
            .select_from(join_expr)
            .where(_status_filter(model, completed=completed, statuses=statuses))
            .where(_time_filter(model, hours=hours))
        )

    if hasattr(model, "tag"):
        tag_field = model.tag.label("tag")
    else:
        tag_field = literal(None).label("tag")

    return (
        select(
            model.id,
            tag_field.label("tag"),
            model.task_status,
            model.created_at,
            model.end_date,
            model.error_msg,
            cast(literal(tag), String).label("task_type"),
        )
        .where(_status_filter(model, completed=completed, statuses=statuses))
        .where(_time_filter(model, hours=hours))
    )


def _status_filter(
    model,  # noqa: ANN001
    *,
    completed: bool | None,
    statuses: set[TaskStatus],
) -> ColumnElement[bool]:
    """Build SQL filter for task completion and status criteria.

    Args:
        model: SQLModel table class to filter.
        completed: Filter by completion status (True/False/None).
        statuses: Set of specific TaskStatus values to include.

    Returns:
        SQLAlchemy filter condition for WHERE clauses.
    """
    if statuses:
        return model.task_status.in_(statuses)

    if completed is True:
        return model.task_status.in_(_COMPLETED)
    if completed is False:
        return model.task_status.in_(_UNFINISHED)

    return true()


def _time_filter(model, *, hours: int) -> ColumnElement[bool]:  # noqa: ANN001
    """Build SQL filter to limit results to recent tasks.

    Args:
        model: SQLModel table class to filter.
        hours: Time window in hours for recent tasks.

    Returns:
        SQLAlchemy filter condition for unfinished or recently completed tasks.
    """
    threshold = datetime.now(tz=UTC) - timedelta(hours=hours)
    return or_(
        model.task_status.in_(_UNFINISHED),
        func.coalesce(model.end_date, func.now()) >= threshold,
    )
