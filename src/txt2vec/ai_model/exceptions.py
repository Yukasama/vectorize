"""Embeddings Exceptions."""

from fastapi import status

from txt2vec.errors import AppError, ErrorCode

__all__ = ["ModelNotFoundError"]


class ModelNotFoundError(AppError):
    """Exception raised when the model is not found."""

    error_code = ErrorCode.NOT_FOUND
    status_code = status.HTTP_404_NOT_FOUND

    def __init__(self, model_id: str) -> None:
        """Initialize with the model ID."""
        super().__init__(f"Model with ID {model_id} not found")
