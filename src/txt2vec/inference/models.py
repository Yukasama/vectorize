"""InferenceCounter model."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlmodel import Column, DateTime, Field, Relationship, SQLModel, func

if TYPE_CHECKING:
    from txt2vec.ai_model.models import AIModel

__all__ = ["InferenceCounter"]


class InferenceCounter(SQLModel, table=True):
    """Inference counter model for tracking model usage."""

    __tablename__ = "inference_counter"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        description="Unique identifier for the inference counter.",
    )

    ai_model_id: uuid.UUID = Field(
        default=None,
        foreign_key="ai_model.id",
        index=True,
        description="ID of the AI model used for inference.",
    )

    ai_model: Optional["AIModel"] = Relationship(
        back_populates="inference_counters",
        sa_relationship_kwargs={"cascade": "all, delete"},
    )

    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True), insert_default=func.now(), index=True
        ),
        description="Timestamp when the inference was recorded.",
    )
