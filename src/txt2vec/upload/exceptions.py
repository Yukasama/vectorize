"""Exception classes for model upload handling."""

from fastapi import status

from txt2vec.common.app_error import AppError
from txt2vec.config.errors import ErrorCode
from txt2vec.datasets.utils.file_size_formatter import format_file_size


class UploadTaskNotFound(AppError):  # noqa: N818
    """Exception raised when no UploadTask is found."""

    error_code = ErrorCode.UPLOAD_TASK_NOT_FOUND
    message = "No task with the provided ID exists."
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class InvalidModelError(AppError):
    """Exception raised when the model file is not a valid PyTorch model."""

    error_code = ErrorCode.INVALID_FILE
    message = "Invalid PyTorch model file"
    status_code = status.HTTP_400_BAD_REQUEST


class EmptyModelError(AppError):
    """Exception raised when the model file is empty."""

    error_code = ErrorCode.EMPTY_FILE
    message = "Empty model file"
    status_code = status.HTTP_400_BAD_REQUEST


class ModelTooLargeError(AppError):
    """Exception raised when the model file is too large."""

    error_code = ErrorCode.INVALID_FILE
    status_code = status.HTTP_400_BAD_REQUEST

    def __init__(self, size: int) -> None:
        """Initialize with the size of the file."""
        formatted_size = format_file_size(size)
        super().__init__(f"Model file is too large: {formatted_size}")


class UnsupportedModelFormatError(AppError):
    """Exception raised when the model file format is not supported."""

    error_code = ErrorCode.UNSUPPORTED_FORMAT
    message = "This model format is not supported. Supported formats: .pt, .pth, .bin"
    status_code = status.HTTP_400_BAD_REQUEST


class InvalidZipError(AppError):
    """Exception raised when a ZIP file is corrupted or invalid."""

    error_code = ErrorCode.INVALID_FILE
    message = "Invalid or corrupted ZIP file"
    status_code = status.HTTP_400_BAD_REQUEST


class NoValidModelsFoundError(AppError):
    """Exception raised when no valid PyTorch models were found in the upload."""

    error_code = ErrorCode.INVALID_FILE
    message = "No valid PyTorch model files found"
    status_code = status.HTTP_400_BAD_REQUEST


class DatabaseError(AppError):
    """Exception raised when there is a database error."""

    error_code = ErrorCode.DATABASE_ERROR
    message = "Database connection was lost"
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class ServiceUnavailableError(AppError):
    """Exception raised when the service is unavailable."""

    error_code = ErrorCode.SERVICE_UNAVAILABLE
    message = "Service is temporarily unavailable"
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


class ModelAlreadyExistsError(AppError):
    """Exception raised when the model already exists in the database."""

    error_code = ErrorCode.MODEL_ALREADY_EXISTS
    status_code = status.HTTP_409_CONFLICT

    def __init__(self, model_tag: str) -> None:
        """Initialize with the model tag."""
        super().__init__(f"Model with tag '{model_tag}' already exists.")
