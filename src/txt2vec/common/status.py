"""Status model for task or upload."""

from enum import StrEnum


class Status(StrEnum):
    """Status of a process."""

    COMPLETED = "C"
    FAILED = "F"
    RUNNING = "R"
    CANCELED = "CA"
