"""ErrorInfo model for process errors."""

import uuid
from datetime import datetime

from sqlmodel import Column, DateTime, Field, SQLModel, func

__all__ = ["ErrorInfo"]


class ErrorInfo(SQLModel, table=True):
    """ErrorInfo model."""

    __tablename__ = "errorinfo"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        description="Unique identifier for the error info.",
    )

    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True),
            insert_default=func.now(),
        ),
        description="Timestamp when the error was recorded.",
    )

    status_code: int = Field(
        ge=100, lt=1000, description="HTTP-like status code representing the error."
    )

    path: str = Field(description="Request path or resource where the error occurred.")
