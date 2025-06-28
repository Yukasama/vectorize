"""Exceptions for task operations."""

from uuid import UUID

from fastapi import status

from vectorize.common.app_error import AppError
from vectorize.config.errors import ErrorCode


class TaskNotFoundError(AppError):
    """Exception raised when the task is not found."""

    error_code = ErrorCode.NOT_FOUND
    status_code = status.HTTP_404_NOT_FOUND

    def __init__(self, task_id: UUID | str) -> None:
        """Initialize with the task ID."""
        super().__init__(f"Task with ID {task_id} not found")
