"""Exception classes for model upload handling."""

from fastapi import status

from txt2vec.datasets.utils import format_file_size  # Nur zur Erinnerung
from txt2vec.errors import AppError, ErrorCode


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
