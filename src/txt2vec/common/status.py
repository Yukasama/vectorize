"""Status model for task or upload."""

from enum import StrEnum

__all__ = ["Status"]


class Status(StrEnum):
    """Status of a process."""

    COMPLETED = "C"
    FAILED = "F"
    RUNNING = "R"
    CANCELED = "CA"
