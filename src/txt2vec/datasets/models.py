"""Dataset models."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlmodel import Column, DateTime, Field, Relationship, SQLModel, func

from txt2vec.synthesis.models import Synthesis

from .classification import Classification

if TYPE_CHECKING:
    from txt2vec.synthesis.models import Synthesis

__all__ = ["Dataset"]


class Dataset(SQLModel, table=True):
    """Dataset model."""

    __tablename__ = "dataset"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    """Unique identifier for the dataset."""

    version: int = Field(default=0)
    """Version number of the dataset."""

    file_name: str = Field(index=True, unique=True)
    """Filename of the dataset file on the storage unit."""

    name: str
    """Name of the dataset."""

    classification: Classification
    """Classification type of the dataset."""

    rows: int
    """Number of rows in the dataset."""

    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True),
            insert_default=func.now(),
        ),
    )
    """Timestamp when the dataset was created."""

    updated_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True),
            onupdate=func.now(),
            insert_default=func.now(),
        ),
    )
    """Timestamp when the dataset was last updated."""

    synthesis_id: uuid.UUID | None = Field(default=None, foreign_key="synthesis.id")
    """Optional ID linking to a synthetic dataset."""

    synthesis: Optional["Synthesis"] = Relationship(back_populates="generated_dataset")
    """Relationship to the associated synthetic dataset."""
