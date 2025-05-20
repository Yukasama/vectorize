"""TaskStatus model for task or upload."""


__all__ = ["TaskStatus"]


from enum import StrEnum


class TaskStatus(StrEnum):
    """Status of a process."""
    PENDING = "P"
    COMPLETED = "C"
    FAILED = "F"
    CANCELED = "CA"
