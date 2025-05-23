"""Dataset models."""

from datetime import datetime
from typing import TYPE_CHECKING, Optional
from uuid import UUID, uuid4

from sqlmodel import Column, DateTime, Field, Relationship, SQLModel, func

from vectorize.synthesis.models import SynthesisTask

from .classification import Classification

if TYPE_CHECKING:
    from vectorize.synthesis.models import SynthesisTask

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
        min_length=1,
        max_length=255,
    )

    rows: int = Field(description="Number of rows in the new dataset", gt=0)

    synthesis_id: UUID | None = Field(
        None, description="Optional ID linking to a synthetic dataset when created"
    )


class DatasetUpdate(SQLModel):
    """Dataset update model with optional fields."""

    name: str = Field(
        description="Name of the dataset to update", min_length=1, max_length=128
    )


class DatasetAll(_DatasetBase):
    """Dataset model for listing datasets with limited fields."""

    id: UUID = Field(description="Unique identifier for the dataset")

    version: int = Field(description="Version number of the dataset")

    rows: int = Field(description="Number of rows in the dataset")

    created_at: datetime = Field(description="Timestamp when the dataset was created")


class DatasetPublic(DatasetAll):
    """Dataset model for detailed view with all fields."""

    updated_at: datetime = Field(
        description="Timestamp when the dataset was last updated"
    )

    synthesis_id: UUID | None = Field(
        None, description="Optional ID linking to a synthetic dataset"
    )


class Dataset(SQLModel, table=True):
    """Dataset model."""

    __tablename__ = "dataset"

    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
        description="Unique identifier for the dataset",
    )

    version: int = Field(default=0, description="Version number of the dataset")

    file_name: str = Field(
        index=True,
        unique=True,
        description="Filename of the dataset file on the storage unit",
        min_length=1,
        max_length=255,
    )

    name: str = Field(
        description="Name of the dataset", index=True, min_length=1, max_length=128
    )

    classification: Classification = Field(
        description="Classification type of the dataset"
    )

    rows: int = Field(description="Number of rows in the dataset", gt=0)

    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), insert_default=func.now()),
        description="Timestamp when the dataset was created",
    )

    updated_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True), onupdate=func.now(), insert_default=func.now()
        ),
        description="Timestamp when the dataset was last updated",
    )

    synthesis_id: UUID | None = Field(
        default=None,
        index=True,
        foreign_key="synthesis_task.id",
        description="Optional ID linking to a synthetic dataset",
    )

    synthesis_task: Optional["SynthesisTask"] = Relationship(
        back_populates="generated_dataset"
    )
