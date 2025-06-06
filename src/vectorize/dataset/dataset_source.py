"""Source types for Dataset."""

from enum import StrEnum

__all__ = ["DatasetSource"]


class DatasetSource(StrEnum):
    """Source types for datasets."""

    HUGGINGFACE = "H"
    LOCAL = "L"
    SYNTHETIC = "S"
