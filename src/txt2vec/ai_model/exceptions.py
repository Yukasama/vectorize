"""AI-Model Exceptions."""

from fastapi import status

from txt2vec.errors import AppError, ErrorCode

__all__ = ["ModelNotFoundError"]


class ModelNotFoundError(AppError):
    """Exception raised when the model is not found."""

    error_code = ErrorCode.NOT_FOUND
    status_code = status.HTTP_404_NOT_FOUND

    def __init__(self, model_id: str) -> None:
        """Initialize with the model ID."""
        super().__init__(f"Model with Model Tag {model_id} not found")


class ModelLoadError(AppError):
    """Exception raised when the model failed to load."""

    error_code = ErrorCode.SERVER_ERROR
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, model_id: str) -> None:
        """Initialize with the model ID."""
        super().__init__(f"Model with Model Tag {model_id} failed to load")
