"""Exception classes for model upload handling."""

from fastapi import status

from txt2vec.common.app_error import AppError
from txt2vec.config.errors import ErrorCode
from txt2vec.datasets.utils.file_size_formatter import format_file_size


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


class InvalidGitHubUrlError(AppError):
    """Fehler beim Zugriff auf die GitHub URL."""

    error_code = ErrorCode.INVALID_URL
    message = "Invalid GitHub URL."
    status_code = status.HTTP_400_BAD_REQUEST


class MissingDownloadUrlError(AppError):
    """Fehlerhafte Antwort der GitHub API."""

    error_code = ErrorCode.SERVICE_UNAVAILABLE
    message = "GitHub API did not return a download URL."
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class ModelFileNotFoundError(AppError):
    """Fehler beim Zugriff auf das Model."""

    error_code = ErrorCode.NOT_FOUND
    message = "Model file not found in the specified repository."
    status_code = status.HTTP_404_NOT_FOUND


class ModelDownloadError(AppError):
    """Fehler beim Zugriff auf das Model."""

    error_code = ErrorCode.SERVICE_UNAVAILABLE
    message = "Error downloading the model from GitHub."
    status_code = status.HTTP_502_BAD_GATEWAY


class GitHubApiError(AppError):
    """Fehler beim Zugriff auf die GitHub Api."""

    error_code = ErrorCode.SERVICE_UNAVAILABLE
    message = "Unexpected error calling the GitHub API."
    status_code = status.HTTP_502_BAD_GATEWAY
