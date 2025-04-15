"""Synthesis model."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlmodel import Field, Relationship, SQLModel, func

from txt2vec.common.models import ErrorInfo
from txt2vec.common.status import Status

if TYPE_CHECKING:
    from txt2vec.datasets.models import Dataset


class Synthesis(SQLModel, table=True):
    """Synthetic generation model."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    """Unique identifier for the synthetic generation."""

    start_date: datetime = Field(sa_column_kwargs={"server_default": func.now()})
    """Start time of the synthetic generation."""

    end_date: datetime | None = Field(default=None)
    """Optional end time of the synthetic generation."""

    error_msg: str | None = Field(default=None)
    """Optional error message encountered during generation."""

    error_id: uuid.UUID | None = Field(default=None, foreign_key="errorinfo.id")
    """Optional foreign key linking to a specific error."""

    error: ErrorInfo | None = Relationship(
        sa_relationship_kwargs={"back_populates": None}
    )
    """Relationship to the associated error information."""

    status: Status = Field(sa_column_kwargs={"server_default": Status.RUNNING})
    """Current status of the synthetic generation."""

    datasets: Optional["Dataset"] = Relationship(back_populates="synthesis")
    """Relationship to the associated datasets."""
