"""Dataset models."""

import uuid
from datetime import datetime

from sqlmodel import Column, DateTime, Field, Relationship, SQLModel, func

from txt2vec.datasets.classification import Classification
from txt2vec.synthesis.models import Synthesis


class Dataset(SQLModel, table=True):
    """Dataset model."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    """Unique identifier for the dataset."""

    name: str = Field(index=True, unique=True)
    """Name of the dataset."""

    classification: Classification
    """Classification type of the dataset."""

    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True),
            server_default=func.now(),
            nullable=True,
        ),
    )
    """Timestamp when the dataset was created."""

    updated_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True),
            onupdate=func.now(),
            nullable=True,
        ),
    )
    """Timestamp when the dataset was last updated (initially NULL)."""

    synthesis_id: uuid.UUID | None = Field(default=None, foreign_key="synthesis.id")
    """Optional ID linking to a synthetic dataset."""

    synthesis: Synthesis | None = Relationship(back_populates="datasets")
    """Relationship to the associated synthetic dataset."""
