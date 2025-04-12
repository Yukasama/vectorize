"""Exceptions for dataset processing."""

from enum import StrEnum

from fastapi import HTTPException, status
from loguru import logger


class ErrorCode(StrEnum):
    """Error codes for standardized error handling."""

    INVALID_FILE = "INVALID_FILE"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    INVALID_CSV_FORMAT = "INVALID_CSV_FORMAT"
    DATASET_NOT_FOUND = "DATASET_NOT_FOUND"
    PROCESSING_ERROR = "PROCESSING_ERROR"
    SERVER_ERROR = "SERVER_ERROR"


class BaseDatasetError(Exception):
    """Base exception class with error code and message."""

    error_code = ErrorCode.SERVER_ERROR
    message = "An unexpected error occurred"
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self):
        """Initialize the exception with a default message."""
        super().__init__(self.message)


class InvalidFileError(BaseDatasetError):
    """Exception raised when the file is invalid."""

    error_code = ErrorCode.INVALID_FILE
    message = "Invalid file format"
    status_code = status.HTTP_400_BAD_REQUEST


class UnsupportedFormatError(BaseDatasetError):
    """Exception raised when the file format is not supported."""

    error_code = ErrorCode.UNSUPPORTED_FORMAT
    message = (
        "This format is not supported. Supported formats: csv, json, xml, xlsx, xls"
    )
    status_code = status.HTTP_400_BAD_REQUEST


class InvalidCSVFormatError(BaseDatasetError):
    """Exception raised when the CSV format is invalid."""

    error_code = ErrorCode.INVALID_CSV_FORMAT
    message = "Invalid CSV format"
    status_code = status.HTTP_400_BAD_REQUEST


class DatasetNotFoundError(BaseDatasetError):
    """Exception raised when the dataset is not found."""

    error_code = ErrorCode.DATASET_NOT_FOUND
    message = "Dataset not found"
    status_code = status.HTTP_404_NOT_FOUND


class ProcessingError(BaseDatasetError):
    """Exception raised when there's an error processing the file."""

    error_code = ErrorCode.PROCESSING_ERROR
    message = "Error processing the dataset"
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


def handle_dataset_exception(exception: Exception) -> HTTPException:
    """Handle exceptions and convert them to HTTPExceptions."""
    if isinstance(exception, HTTPException):
        return exception

    if isinstance(exception, BaseDatasetError):
        logger.warning(
            f"Dataset exception: {exception.error_code} - {exception.message}",
        )
        return HTTPException(
            status_code=exception.status_code,
            detail={"code": exception.error_code, "message": exception.message},
        )

    logger.error(f"Unhandled exception: {type(exception).__name__} - {exception!s}")
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail={
            "code": ErrorCode.SERVER_ERROR,
            "message": "An unexpected server error occurred",
        },
    )
