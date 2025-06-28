"""Task type definitions."""

from enum import StrEnum

__all__ = ["TaskType"]


class TaskType(StrEnum):
    """Task type enumeration for different task categories."""

    MODEL_UPLOAD = "model_upload"
    SYNTHESIS = "synthesis"
    DATASET_UPLOAD = "dataset_upload"
    TRAINING = "training"
    EVALUATION = "evaluation"
