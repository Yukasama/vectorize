"""Model for action responses."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel

from vectorize.common.task_status import TaskStatus

__all__ = ["TasksModel"]


class TasksModel(BaseModel):
    """Pydantic model for action responses."""

    id: UUID
    tag: str
    task_status: TaskStatus
    created_at: datetime
    end_date: datetime | None
    error_msg: str | None = None
    task_type: Literal["model_upload", "synthesis", "dataset_upload"]
