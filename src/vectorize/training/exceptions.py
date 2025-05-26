"""Training exceptions."""

from fastapi import status

from vectorize.common.app_error import AppError
from vectorize.config.errors import ErrorCode

__all__ = [
    "TrainingDatasetNotFoundError",
    "TrainingModelNotFoundError",
    "TrainingModelWeightsNotFoundError",
]


class TrainingDatasetNotFoundError(AppError):
    """Exception raised when the training dataset file is not found."""

    error_code = ErrorCode.NOT_FOUND
    status_code = status.HTTP_404_NOT_FOUND

    def __init__(self, dataset_path: str) -> None:
        """Initialize with the dataset file path."""
        super().__init__(f"Training dataset file not found: {dataset_path}")


class TrainingModelNotFoundError(AppError):
    """Exception raised when the training model directory is not found or invalid."""

    error_code = ErrorCode.NOT_FOUND
    status_code = status.HTTP_404_NOT_FOUND

    def __init__(self, model_path: str) -> None:
        """Initialize with the model directory path."""
        super().__init__(f"Training model directory not found or invalid: {model_path}")


class TrainingModelWeightsNotFoundError(AppError):
    """Exception raised when model weights (.bin/.safetensors) are missing or invalid."""

    error_code = ErrorCode.NOT_FOUND
    status_code = status.HTTP_404_NOT_FOUND

    def __init__(self, model_path: str) -> None:
        super().__init__(f"Model weights (.bin/.safetensors) not found or invalid in: {model_path}")
