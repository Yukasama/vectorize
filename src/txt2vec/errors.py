"""Central exception handling for the application."""

from enum import StrEnum

from fastapi import status

__all__ = ["AppError", "ErrorCode"]


class ErrorCode(StrEnum):
    """Error codes for standardized error handling."""

    # General errors
    SERVER_ERROR = "SERVER_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"

    # Dataset errors
    INVALID_FILE = "INVALID_FILE"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    INVALID_CSV_FORMAT = "INVALID_CSV_FORMAT"
    EMPTY_FILE = "EMPTY_FILE"
    PROCESSING_ERROR = "PROCESSING_ERROR"


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
