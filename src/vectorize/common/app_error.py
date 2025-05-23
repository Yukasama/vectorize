"""Generic application errors."""

from fastapi import status

from vectorize.config.errors import ErrorCode

__all__ = ["AppError", "ETagError"]


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


class ETagError(Exception):
    """Base exception for application errors."""

    error_code = ErrorCode.VERSION_MISMATCH
    message = "ETag mismatch"
    status_code = status.HTTP_412_PRECONDITION_FAILED
    version: str = '"0"'

    def __init__(self, version: str, message: str | None = None) -> None:
        """Initialize with optional custom message."""
        if message:
            self.message = message
        self.version = version

        super().__init__(self.version, self.message)
