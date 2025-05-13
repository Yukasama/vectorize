"""AI-Model Exceptions."""

from typing import Optional
from fastapi import status

from txt2vec.common.app_error import AppError
from txt2vec.config.errors import ErrorCode

__all__ = ["ModelLoadError", "ModelNotFoundError", "UnsupportedModelError"]


class ModelNotFoundError(AppError):
    """Exception raised when the model is not found."""

    error_code = ErrorCode.NOT_FOUND
    status_code = status.HTTP_404_NOT_FOUND

    def __init__(self, model_id: str, model_tag: Optional[str]) -> None:
        """Initialize with the model tag."""

        if model_tag:
            msg = f"Model with id {model_id} and tag {model_tag} not found"
        else:
            msg = f"Model with tag {model_id} not found"
        super().__init__(msg)

class ModelLoadError(AppError):
    """Exception raised when the model failed to load."""

    error_code = ErrorCode.SERVER_ERROR
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, model_id: str) -> None:
        """Initialize with the model tag."""
        super().__init__(f"Model with tag {model_id} failed to load")


class UnsupportedModelError(AppError):
    """Exception raised when the model is not supported."""

    error_code = ErrorCode.UNSUPPORTED_FORMAT
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY

    def __init__(self, model: str) -> None:
        """Initialize with the model format."""
        super().__init__(f"Model format {model} not supported")
