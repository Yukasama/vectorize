"""Source types for AIModel."""

from enum import StrEnum

__all__ = ["ModelSource"]


class ModelSource(StrEnum):
    """Source types for AI models."""

    GITHUB = "GitHub"
    HUGGINGFACE = "HuggingFace"
    LOCAL = "Local"


class RemoteModelSource(StrEnum):
    """Source types for AI models from remote sources."""

    GITHUB = "GitHub"
    HUGGINGFACE = "HuggingFace"
