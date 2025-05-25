"""Training exceptions."""

from fastapi import status

from vectorize.common.app_error import AppError
from vectorize.config.errors import ErrorCode

__all__ = ["TrainingDatasetNotFoundError"]


class TrainingDatasetNotFoundError(AppError):
    """Exception raised when the training dataset file is not found."""

    error_code = ErrorCode.NOT_FOUND
    status_code = status.HTTP_404_NOT_FOUND

    def __init__(self, dataset_path: str) -> None:
        """Initialize with the dataset file path."""
        super().__init__(f"Training dataset file not found: {dataset_path}")
