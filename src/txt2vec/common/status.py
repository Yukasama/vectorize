"""TaskStatus model for task or upload."""

from enum import Enum

__all__ = ["TaskStatus"]


class TaskStatus(str, Enum):
    """Status of a process."""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
