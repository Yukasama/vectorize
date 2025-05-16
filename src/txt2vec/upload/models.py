"""TODO."""
import uuid
from datetime import datetime

from sqlalchemy.types import Enum as SQLEnum
from sqlmodel import Column, DateTime, Field, SQLModel, func

from txt2vec.ai_model.model_source import ModelSource
from txt2vec.common.status import TaskStatus


class UploadTask(SQLModel, table=True):
    """Upload process."""
    __tablename__ = "upload_task"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        description="Unique identifier for the upload task.",
    )

    model_tag: str = Field(
        description="Tag of the model file being uploaded.",
        min_length=1,
        max_length=128,
    )

    task_status: TaskStatus = Field(
        default=TaskStatus.PENDING,
        sa_column=Column(
            SQLEnum(TaskStatus),  # <-- SQLAlchemy Enum, not Python StrEnum
            nullable=False
        ),
        description="Status of the upload task.",
    )

    source: ModelSource = Field(
        default=ModelSource.GITHUB,
        sa_column=Column(
            SQLEnum(ModelSource),  # <-- same here
            nullable=False
        ),
        description="Source of the model (github or huggingface).",
    )

    end_date: datetime | None = Field(
        default=None,
        description="Optional end time of the upload task.",
    )

    error_msg: str | None = Field(
        default=None,
        description="Optional error message encountered during upload.",
    )

# pylint: disable=not-callable
    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True),
            insert_default=func.now(),
        ),
        description="Timestamp when the upload task was created.",
    )
# pylint: disable=not-callable
    updated_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True),
            insert_default=func.now(),
            onupdate=func.now(),
        ),
        description="Timestamp when the upload task was last updated.",
    )
