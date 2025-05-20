"""TaskStatus model for task or upload."""

from enum import StrEnum

__all__ = ["TaskStatus"]


class TaskStatus(StrEnum):
    """Status of a process."""

    PENDING = "Pending"
    DONE = "Done"
    FAILED = "Failed"
    CANCELED = "Canceled"
