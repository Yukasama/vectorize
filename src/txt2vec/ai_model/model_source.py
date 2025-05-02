"""Source types for AI models."""

from enum import StrEnum

__all__ = ["ModelSource"]


class ModelSource(StrEnum):
    """Source types for AI models."""

    GITHUB = "G"
    HUGGINGFACE = "H"
    LOCAL = "L"
