"""Enum to represent Github and HuggingFace uploads."""
from enum import Enum


class ModelSource(str, Enum):
    """Enum representing the possible sources from which a model can be uploaded.

    Attributes:
        GITHUB: Indicates the model is sourced from GitHub.
        HUGGINGFACE: Indicates the model is sourced from HuggingFace.
    """
    GITHUB = "github"
    HUGGINGFACE = "huggingface"
