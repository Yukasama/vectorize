"""ErrorInfo model for process errors."""

import uuid
from datetime import datetime

from sqlmodel import Field, SQLModel, func


class ErrorInfo(SQLModel, table=True):
    """ErrorInfo model."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    """Unique identifier for the error info."""

    created_at: datetime | None = Field(
        default=None, sa_column_kwargs={"server_default": func.now()}
    )
    """Timestamp when the error was recorded."""

    status_code: int = Field(ge=100, lt=1000)
    """HTTP-like status code representing the error."""

    path: str
    """Request path or resource where the error occurred."""
