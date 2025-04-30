"""Common exceptions."""

from fastapi import status

from txt2vec.config.errors import ErrorCode

__all__ = [
    "InternalServerError",
    "NotFoundError",
]


class NotFoundError(Exception):
    """Exception raised when something is not found."""

    error_code = ErrorCode.NOT_FOUND
    message = "Resource not found"
    status_code = status.HTTP_404_NOT_FOUND


class InternalServerError(Exception):
    """Exception raised for internal server errors."""

    error_code = ErrorCode.SERVER_ERROR
    message = "Internal server error"
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
