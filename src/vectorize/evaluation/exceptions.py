"""Evaluation exceptions."""

from fastapi import status

from vectorize.common.app_error import AppError
from vectorize.config.errors import ErrorCode

__all__ = ["EvaluationTaskNotFoundError", "InvalidDatasetIdError"]


class EvaluationTaskNotFoundError(AppError):
    """Raised when evaluation task is not found."""

    def __init__(self, task_id: str) -> None:
        """Initialize exception.

        Args:
            task_id: The task ID that was not found
        """
        super().__init__(f"Evaluation task not found: {task_id}")
        self.error_code = ErrorCode.NOT_FOUND
        self.status_code = status.HTTP_404_NOT_FOUND


class InvalidDatasetIdError(AppError):
    """Exception raised when a dataset_id is not a valid UUID."""

    error_code = ErrorCode.NOT_FOUND
    status_code = status.HTTP_404_NOT_FOUND

    def __init__(self, dataset_id: str) -> None:
        """Initialize with the invalid dataset ID."""
        super().__init__(f"Dataset ID is not a valid UUID: {dataset_id}")
