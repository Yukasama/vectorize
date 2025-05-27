"""Common exceptions."""

from uuid import UUID

from fastapi import status

from vectorize.common.app_error import AppError, ETagError
from vectorize.config.errors import ErrorCode

__all__ = ["InternalServerError", "NotFoundError"]


class NotFoundError(AppError):
    """Exception raised when something is not found."""

    error_code = ErrorCode.NOT_FOUND
    message = "Resource not found"
    status_code = status.HTTP_404_NOT_FOUND


class InternalServerError(AppError):
    """Exception raised for internal server errors."""

    error_code = ErrorCode.SERVER_ERROR
    message = "Internal server error"
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class VersionMismatchError(ETagError):
    """Exception raised when the dataset version does not match."""

    error_code = ErrorCode.VERSION_MISMATCH
    status_code = status.HTTP_412_PRECONDITION_FAILED

    def __init__(self, resource_id: str | int | UUID, version: int) -> None:
        """Initialize with the resource ID and version."""
        super().__init__(
            str(version), f"Resource with ID {resource_id} has version {version}"
        )


class VersionMissingError(AppError):
    """Exception raised when the dataset version is missing in the request."""

    error_code = ErrorCode.VERSION_MISSING
    status_code = status.HTTP_428_PRECONDITION_REQUIRED

    def __init__(self, resource_id: str | int | UUID) -> None:
        """Initialize with the resource ID."""
        super().__init__(
            f"If-Match header required for updating resource {resource_id}"
        )
