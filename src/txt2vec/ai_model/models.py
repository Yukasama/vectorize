"""AI-Model model."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlmodel import Column, DateTime, Field, Relationship, SQLModel, func

from .model_source import ModelSource

if TYPE_CHECKING:
    from txt2vec.inference.models import InferenceCounter

__all__ = ["AIModel"]


class AIModel(SQLModel, table=True):
    """AIModel model."""

    __tablename__ = "ai_model"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        description="Unique identifier for the AI model.",
    )

    version: int = Field(default=0, description="Version number of the AI model.")

    name: str = Field(description="Name of the AI model.")

    model_tag: str = Field(
        index=True, unique=True, description="Tag of the AI model file."
    )

    source: ModelSource = Field(
        description="Source of the model (github, huggingface, or local)."
    )

    inference_counters: list["InferenceCounter"] = Relationship(
        back_populates="ai_model",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )

    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True),
            insert_default=func.now(),
        ),
        description="Timestamp when the AI model was created.",
    )

    updated_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True),
            onupdate=func.now(),
            insert_default=func.now(),
        ),
        description="Timestamp when the AI model was last updated.",
    )
