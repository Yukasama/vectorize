"""Common exceptions."""

from fastapi import status

from txt2vec.common.app_error import AppError, ETagError
from txt2vec.config.errors import ErrorCode

__all__ = [
    "InternalServerError",
    "NotFoundError",
]


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

    def __init__(self, dataset_id: str, version: int) -> None:
        """Initialize with the dataset ID and version."""
        super().__init__(version, f"Dataset with ID {dataset_id} has version {version}")


class VersionMissingError(ETagError):
    """Exception raised when the dataset version is missing in the request."""

    error_code = ErrorCode.VERSION_MISSING
    status_code = status.HTTP_428_PRECONDITION_REQUIRED

    def __init__(self, dataset_id: str, version: int) -> None:
        """Initialize with the dataset ID and version."""
        self.dataset_id = dataset_id
        super().__init__(
            version, f"If-Match header required for updating dataset {dataset_id}"
        )
