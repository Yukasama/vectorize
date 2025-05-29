"""Training exceptions."""

from fastapi import status

from vectorize.common.app_error import AppError
from vectorize.config.errors import ErrorCode

__all__ = [
    "EmptyDatasetListError",
    "InvalidBatchSizeError",
    "InvalidEpochsError",
    "InvalidLearningRateError",
    "TrainingDatasetNotFoundError",
    "TrainingModelNotFoundError",
    "TrainingModelWeightsNotFoundError",
    "TrainingTaskNotFoundError",
]


class TrainingDatasetNotFoundError(AppError):
    """Exception raised when the training dataset file is not found."""

    error_code = ErrorCode.NOT_FOUND
    status_code = status.HTTP_404_NOT_FOUND

    def __init__(self, dataset_path: str) -> None:
        """Initialize with the dataset file path."""
        super().__init__(f"Training dataset file not found: {dataset_path}")


class TrainingModelNotFoundError(AppError):
    """Exception raised when the training model directory is not found or invalid."""

    error_code = ErrorCode.NOT_FOUND
    status_code = status.HTTP_404_NOT_FOUND

    def __init__(self, model_path: str) -> None:
        """Initialize with the model directory path."""
        super().__init__(f"Training model directory not found or invalid: {model_path}")


class TrainingModelWeightsNotFoundError(AppError):
    """Exception raised when model weights (.bin/.safetensors) are invalid."""

    error_code = ErrorCode.NOT_FOUND
    status_code = status.HTTP_404_NOT_FOUND

    def __init__(self, model_path: str) -> None:
        """Initialize the exception for missing or invalid model weights."""
        super().__init__(
            f"Model weights (.bin/.safetensors) not found or invalid in: {model_path}"
        )


class EmptyDatasetListError(AppError):
    """Exception raised when the dataset_paths list is empty for training."""

    error_code = ErrorCode.EMPTY_FILE
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY

    def __init__(self) -> None:
        """Initialize the exception for an empty dataset list."""
        super().__init__("No dataset files provided: dataset_paths list is empty.")


class InvalidEpochsError(AppError):
    """Exception raised when the number of epochs is not positive."""

    error_code = ErrorCode.INVALID_EPOCHS
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY

    def __init__(self, value: int) -> None:
        """Initialize the exception for an invalid number of epochs."""
        super().__init__(f"Number of epochs must be positive (got {value}).")


class InvalidBatchSizeError(AppError):
    """Exception raised when the batch size is not positive."""

    error_code = ErrorCode.INVALID_BATCH_SIZE
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY

    def __init__(self, value: int) -> None:
        """Initialize the exception for an invalid batch size."""
        super().__init__(f"Batch size must be positive (got {value}).")


class InvalidLearningRateError(AppError):
    """Exception raised when the learning rate is not positive."""

    error_code = ErrorCode.INVALID_LEARNING_RATE
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY

    def __init__(self, value: float) -> None:
        """Initialize the exception for an invalid learning rate."""
        super().__init__(f"Learning rate must be positive (got {value}).")


class TrainingTaskNotFoundError(AppError):
    """Exception raised when a TrainingTask with the given ID is not found."""

    error_code = ErrorCode.NOT_FOUND
    status_code = status.HTTP_404_NOT_FOUND

    def __init__(self, task_id: str) -> None:
        """Initialize with the missing training task ID."""
        super().__init__(f"Training task not found: {task_id}")
