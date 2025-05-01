"""Dataset models."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlmodel import Column, DateTime, Field, Relationship, SQLModel, func

from txt2vec.synthesis.models import Synthesis

from .classification import Classification

if TYPE_CHECKING:
    from txt2vec.synthesis.models import Synthesis

__all__ = [
    "Dataset",
    "DatasetAll",
    "DatasetCreate",
    "DatasetPublic",
    "DatasetUpdate",
]


class _DatasetBase(SQLModel):
    """Base Dataset model."""

    name: str = Field(description="Name of the dataset")

    classification: Classification = Field(
        description="Classification type of the dataset"
    )


class DatasetCreate(_DatasetBase):
    """Dataset creation model."""

    file_name: str = Field(
        description="Filename of the dataset file on the storage unit",
    )

    rows: int = Field(description="Number of rows in the dataset")

    synthesis_id: uuid.UUID | None = Field(
        None, description="Optional ID linking to a synthetic dataset"
    )


class DatasetUpdate(SQLModel):
    """Dataset update model with optional fields."""

    name: str | None = Field(None, description="Name of the dataset")


class DatasetAll(_DatasetBase):
    """Dataset model for listing datasets with limited fields."""

    id: uuid.UUID = Field(description="Unique identifier for the dataset")

    rows: int = Field(description="Number of rows in the dataset")

    created_at: datetime | None = Field(
        None, description="Timestamp when the dataset was created"
    )


class DatasetPublic(DatasetAll):
    """Dataset model for detailed view with all fields."""

    updated_at: datetime | None = Field(
        None, description="Timestamp when the dataset was last updated"
    )

    synthesis_id: uuid.UUID | None = Field(
        None, description="Optional ID linking to a synthetic dataset"
    )


class Dataset(SQLModel, table=True):
    """Dataset model."""

    __tablename__ = "dataset"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        description="Unique identifier for the dataset",
    )

    version: int = Field(default=0, description="Version number of the dataset")

    file_name: str = Field(
        index=True,
        unique=True,
        description="Filename of the dataset file on the storage unit",
    )

    name: str = Field(description="Name of the dataset")

    classification: Classification = Field(
        description="Classification type of the dataset"
    )

    rows: int = Field(description="Number of rows in the dataset")

    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True),
            insert_default=func.now(),
        ),
        description="Timestamp when the dataset was created",
    )

    updated_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True),
            onupdate=func.now(),
            insert_default=func.now(),
        ),
        description="Timestamp when the dataset was last updated",
    )

    synthesis_id: uuid.UUID | None = Field(
        default=None,
        foreign_key="synthesis.id",
        description="Optional ID linking to a synthetic dataset",
    )

    synthesis: Optional["Synthesis"] = Relationship(back_populates="generated_dataset")
