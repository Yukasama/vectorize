"""AI-Model model."""

import uuid
from datetime import datetime

from sqlmodel import Column, DateTime, Field, SQLModel, func

from .model_source import ModelSource

__all__ = ["AIModel"]


class AIModel(SQLModel, table=True):
    """AIModel model."""

    __tablename__ = "ai_model"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    """Unique identifier for the AI model."""

    version: int = Field(default=0)
    """Version number of the AI model."""

    model_tag: str = Field(index=True, unique=True)
    """Tag of the AI model file."""

    source: ModelSource
    """Source of the model (github, huggingface, or local)."""

    name: str
    """Name of the AI model."""

    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True),
            insert_default=func.now(),
        ),
    )
    """Timestamp when the AI model was created."""

    updated_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True),
            onupdate=func.now(),
            insert_default=func.now(),
        ),
    )
    """Timestamp when the AI model was last updated."""
