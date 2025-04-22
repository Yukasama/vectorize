"""Dataset exceptions."""

from fastapi import status

from txt2vec.config.config import allowed_extensions
from txt2vec.datasets.utils import format_file_size
from txt2vec.errors import AppError, ErrorCode

__all__ = [
    "DatasetNotFoundError",
    "InvalidCSVFormatError",
    "InvalidFileError",
    "ProcessingError",
    "UnsupportedFormatError",
]


class InvalidFileError(AppError):
    """Exception raised when the file is invalid."""

    error_code = ErrorCode.INVALID_FILE
    message = "Invalid file format"
    status_code = status.HTTP_400_BAD_REQUEST


class FileTooLargeError(AppError):
    """Exception raised when the file is too large."""

    error_code = ErrorCode.INVALID_FILE
    status_code = status.HTTP_400_BAD_REQUEST

    def __init__(self, size: int) -> None:
        """Initialize with the size of the file."""
        formatted_size = format_file_size(size)
        super().__init__(f"File is too large: {formatted_size}")


class UnsupportedFormatError(AppError):
    """Exception raised when the file format is not supported."""

    error_code = ErrorCode.UNSUPPORTED_FORMAT
    message = (
        "This format is not supported. Supported formats: "
        f"{', '.join(allowed_extensions)}"
    )
    status_code = status.HTTP_400_BAD_REQUEST


class InvalidCSVFormatError(AppError):
    """Exception raised when the CSV format is invalid."""

    error_code = ErrorCode.INVALID_CSV_FORMAT
    message = (
        "Invalid CSV format, expected: 'id, anchor, positive, negative' as columns"
    )
    status_code = status.HTTP_400_BAD_REQUEST


class EmptyFileError(AppError):
    """Exception raised when the CSV format is empty."""

    error_code = ErrorCode.EMPTY_FILE
    message = "CSV is empty"
    status_code = status.HTTP_400_BAD_REQUEST


class DatasetNotFoundError(AppError):
    """Exception raised when the dataset is not found."""

    error_code = ErrorCode.NOT_FOUND
    message = "Dataset not found"
    status_code = status.HTTP_404_NOT_FOUND

    def __init__(self, dataset_id: str) -> None:
        """Initialize with the dataset ID."""
        super().__init__(f"Dataset with ID {dataset_id} not found")


class ProcessingError(AppError):
    """Exception raised when there's an error processing the file."""

    error_code = ErrorCode.PROCESSING_ERROR
    message = "Error processing the dataset"
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
