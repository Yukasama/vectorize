"""Dataset exceptions."""

from uuid import UUID

from fastapi import status

from vectorize.common.app_error import AppError
from vectorize.config import settings
from vectorize.config.errors import ErrorCode
from vectorize.utils.file_size_formatter import format_file_size

__all__ = [
    "DatasetNotFoundError",
    "EmptyFileError",
    "FileTooLargeError",
    "InvalidCSVColumnError",
    "InvalidCSVFormatError",
    "MissingColumnError",
    "TooManyFilesError",
    "UnsupportedFormatError",
]


_ALLOWED_EXTENSIONS = ", ".join(settings.allowed_extensions)


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
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY

    def __init__(self, file_format: str) -> None:
        """Initialize with the size of the file."""
        super().__init__(
            f"Format '{file_format}' is not supported. Supported formats: "
            f"{_ALLOWED_EXTENSIONS}"
        )


class InvalidCSVFormatError(AppError):
    """Exception raised when the CSV format is invalid."""

    error_code = ErrorCode.INVALID_CSV_FORMAT
    message = "Invalid CSV format, expected: 'question, positive, negative' as columns"
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY


class MissingColumnError(AppError):
    """Exception raised when the CSV format is invalid."""

    error_code = ErrorCode.INVALID_CSV_FORMAT
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY

    def __init__(self, missing_column: str) -> None:
        """Initialize with the column name."""
        super().__init__(f"Column '{missing_column}' is missing in the dataset")


class InvalidCSVColumnError(AppError):
    """Exception raised when the a specified column does not exist."""

    error_code = ErrorCode.INVALID_CSV_FORMAT
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY

    def __init__(self, column_name: str) -> None:
        """Initialize with the column name."""
        super().__init__(f"Column with name '{column_name}' not found in the dataset")


class UnsupportedHuggingFaceFormatError(AppError):
    """Exception raised when the Hugging Face dataset format is unsupported."""

    error_code = ErrorCode.HUGGINGFACE_DATASET_FORMAT_ERROR
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY

    def __init__(self, column_names: list[str]) -> None:
        """Initialize with the column names."""
        super().__init__(f"Dataset format '{column_names}' not supported")


class EmptyFileError(AppError):
    """Exception raised when the CSV format is empty."""

    error_code = ErrorCode.EMPTY_FILE
    message = "CSV is empty"
    status_code = status.HTTP_400_BAD_REQUEST


class DatasetNotFoundError(AppError):
    """Exception raised when the dataset is not found."""

    error_code = ErrorCode.NOT_FOUND
    status_code = status.HTTP_404_NOT_FOUND

    def __init__(self, dataset_id: UUID | str) -> None:
        """Initialize with the dataset ID."""
        super().__init__(f"Dataset with ID {dataset_id} not found")


class DatasetAlreadyExistsError(AppError):
    """Exception raised when the dataset already exists on Hugging Face Hub."""

    error_code = ErrorCode.DATASET_ALREADY_EXISTS
    status_code = status.HTTP_409_CONFLICT

    def __init__(self, dataset_tag: str) -> None:
        """Initialize with the dataset tag."""
        super().__init__(f"Dataset with tag {dataset_tag} already exists in database")


class DatasetIsAlreadyBeingUploadedError(AppError):
    """Exception raised when the dataset is already being uploaded."""

    error_code = ErrorCode.DATASET_ALREADY_EXISTS
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY

    def __init__(self, dataset_tag: str) -> None:
        """Initialize with the dataset tag."""
        super().__init__(f"Dataset with tag {dataset_tag} is already being uploaded")


class HuggingFaceDatasetNotFoundError(AppError):
    """Exception raised when the dataset is not found on Hugging Face Hub."""

    error_code = ErrorCode.NOT_FOUND
    status_code = status.HTTP_404_NOT_FOUND

    def __init__(self, dataset_tag: str) -> None:
        """Initialize with the dataset tag."""
        super().__init__(
            f"Dataset with tag {dataset_tag} not found on Hugging Face Hub"
        )


class TooManyFilesError(AppError):
    """Exception raised when the zip file is too long."""

    _MAX_LENGTH = 5000
    error_code = ErrorCode.INVALID_FILE
    status_code = status.HTTP_400_BAD_REQUEST

    def __init__(self, size: int) -> None:
        """Initialize with the size of the file."""
        formatted_size = format_file_size(size)
        super().__init__(
            f"Zip file is too large: {formatted_size}. Max: {self._MAX_LENGTH} files"
        )


class InvalidXMLFormatError(AppError):
    """Exception raised when the XML format is invalid."""

    error_code = ErrorCode.INVALID_FILE
    message = "Invalid XML format"
    status_code = status.HTTP_400_BAD_REQUEST

    def __init__(self) -> None:
        """Initialize with the error details."""
        super().__init__("XML format error, root element is missing or empty")
