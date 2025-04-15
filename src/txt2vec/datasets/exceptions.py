"""Dataset exceptions."""

from fastapi import status

from txt2vec.handle_exceptions import AppError, ErrorCode

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


class UnsupportedFormatError(AppError):
    """Exception raised when the file format is not supported."""

    error_code = ErrorCode.UNSUPPORTED_FORMAT
    message = (
        "This format is not supported. Supported formats: csv, json, xml, xlsx, xls"
    )
    status_code = status.HTTP_400_BAD_REQUEST


class InvalidCSVFormatError(AppError):
    """Exception raised when the CSV format is invalid."""

    error_code = ErrorCode.INVALID_CSV_FORMAT
    message = "Invalid CSV format"
    status_code = status.HTTP_400_BAD_REQUEST


class EmptyCSVError(AppError):
    """Exception raised when the CSV format is empty."""

    error_code = ErrorCode.EMPTY_CSV
    message = "CSV is empty"
    status_code = status.HTTP_400_BAD_REQUEST


class DatasetNotFoundError(AppError):
    """Exception raised when the dataset is not found."""

    error_code = ErrorCode.DATASET_NOT_FOUND
    message = "Dataset not found"
    status_code = status.HTTP_404_NOT_FOUND


class ProcessingError(AppError):
    """Exception raised when there's an error processing the file."""

    error_code = ErrorCode.PROCESSING_ERROR
    message = "Error processing the dataset"
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
