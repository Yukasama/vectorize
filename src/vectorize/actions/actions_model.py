"""Model for action responses."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel

from vectorize.common.task_status import TaskStatus
from vectorize.dataset.task_model import UploadDatasetTask
from vectorize.synthesis.models import SynthesisTask
from vectorize.upload.models import UploadTask

__all__ = ["ActionModel", "ModelType"]


ModelType = type[UploadDatasetTask | UploadTask | SynthesisTask]


class ActionModel(BaseModel):
    """Pydantic model for action responses."""

    id: UUID
    task_status: TaskStatus
    created_at: datetime
    end_date: datetime | None
    task_type: Literal["model_upload", "synthesis", "dataset_upload"]
