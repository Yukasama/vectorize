"""Generic application error handling module."""

from fastapi import status

from txt2vec.config.errors import ErrorCode

__all__ = ["AppError"]


class AppError(Exception):
    """Base exception for application errors."""

    error_code = ErrorCode.SERVER_ERROR
    message = "An unexpected error occurred"
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, message: str | None = None) -> None:
        """Initialize with optional custom message."""
        if message:
            self.message = message
        super().__init__(self.message)
