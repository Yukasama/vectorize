"""Synthesis model."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlmodel import Column, DateTime, Field, Relationship, SQLModel, func

from txt2vec.common.models import ErrorInfo
from txt2vec.common.status import Status

if TYPE_CHECKING:
    from txt2vec.datasets.models import Dataset

__all__ = ["Synthesis"]


class Synthesis(SQLModel, table=True):
    """Synthetic generation model."""

    __tablename__ = "synthesis"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    """Unique identifier for the synthetic generation."""

    start_time: datetime = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True),
            insert_default=func.now(),
        ),
    )
    """Start time of the synthetic generation."""

    end_date: datetime | None = Field(default=None)
    """Optional end time of the synthetic generation."""

    error_msg: str | None = Field(default=None)
    """Optional error message encountered during generation."""

    error_id: uuid.UUID | None = Field(default=None, foreign_key="errorinfo.id")
    """Optional foreign key linking to a specific error."""

    error: ErrorInfo | None = Relationship(back_populates=None)
    """Relationship to the associated error information."""

    status: Status = Field(sa_column=Column(server_default=Status.RUNNING))
    """Current status of the synthetic generation."""

    generated_dataset: Optional["Dataset"] = Relationship(back_populates="synthesis")
    """Relationship to the associated datasets."""

    deleted_at: datetime | None = Field(default=None)
    """Optional timestamp for when the synthetic generation was deleted."""
