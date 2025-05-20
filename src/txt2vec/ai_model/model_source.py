"""Source types for AIModel."""

from enum import StrEnum

__all__ = ["ModelSource"]


class ModelSource(StrEnum):
    """Source types for AI models."""

    GITHUB = "GitHub"
    HUGGINGFACE = "HuggingFace"
    LOCAL = "Local"


class ModelSourceAsync(StrEnum):
    """Source types for AI models that upload asynchronously."""

    GITHUB = "GitHub"
    HUGGINGFACE = "HuggingFace"
