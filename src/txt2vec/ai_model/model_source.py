"""Source types for AIModel."""

from enum import StrEnum

__all__ = ["ModelSource"]


class ModelSource(StrEnum):
    """Source types for AI models."""

    GITHUB = "G"
    HUGGINGFACE = "H"
    LOCAL = "L"


class RemoteModelSource(StrEnum):
    """Source types for AI models from remote sources."""

    GITHUB = "G"
    HUGGINGFACE = "H"
