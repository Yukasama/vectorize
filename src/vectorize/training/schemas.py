"""Schemas for the training API."""

from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from vectorize.training.models import TrainingTask

from .exceptions import (
    EmptyDatasetListError,
    InvalidBatchSizeError,
    InvalidLearningRateError,
)


class TrainConfig(BaseModel):
    """Hyperparameters for model training."""

    epochs: int = Field(1, description="Number of training epochs", gt=0)
    learning_rate: float = Field(5e-5, description="Learning rate for training")
    per_device_train_batch_size: int = Field(8, description="Batch size per device")


class TrainRequest(BaseModel):
    """Request body for model training."""

    model_path: str = Field(
        description=(
            "Path to the local model directory (no Huggingface download allowed)"
        ),
    )
    dataset_paths: list[str] = Field(description="Paths to the training datasets (local, multiple allowed)"
    )
    output_dir: str = Field(description="Path to save the trained model")
    epochs: int = Field(1, description="Number of training epochs", gt=0)
    learning_rate: float = Field(5e-5, description="Learning rate for training")
    per_device_train_batch_size: int = Field(8, description="Batch size per device")

    @field_validator("dataset_paths")
    @classmethod
    def check_paths(cls, paths: list[str]) -> list[str]:
        """Validate that all dataset paths end with .csv and list is not empty."""
        if not paths:
            raise EmptyDatasetListError()
        for p in paths:
            if not p.endswith(".csv"):
                raise ValueError("All dataset paths must be .csv files")
        return paths

    @field_validator("model_path")
    @classmethod
    def check_model_path(cls, path: str) -> str:
        """Validate that the model path exists and is a directory."""
        if not Path(path).is_dir():
            raise ValueError(
                f"Model path does not exist or is not a directory: {path}"
            )
        return path

 #  Nutzer soll nicht den Pfad, sondern die ID angeben.
 #  Im Router wird dann der Pfad ermittelt,
 #  aber der Router muss auf den Service zugreifen können.
 #  Field Validator müssen alle raus, die brauche ich nicht, weil Pydantic automatisch validiert.

    @field_validator("per_device_train_batch_size")
    @classmethod
    def check_batch_size(cls, value: int) -> int:
        """Raise InvalidBatchSizeError if batch size is not positive."""
        if value <= 0:
            raise InvalidBatchSizeError(value)
        return value

    @field_validator("learning_rate")
    @classmethod
    def check_learning_rate(cls, value: float) -> float:
        """Raise InvalidLearningRateError if learning rate is not positive."""
        if value <= 0:
            raise InvalidLearningRateError(value)
        return value


class TrainingStatusResponse(BaseModel):
    task_id: str
    status: str
    created_at: str | None = None
    end_date: str | None = None
    error_msg: str | None = None
    trained_model_id: str | None = None

    @classmethod
    def from_task(cls, task: TrainingTask) -> "TrainingStatusResponse":
        return cls(
            task_id=str(task.id),
            status=task.task_status.name,
            created_at=task.created_at.isoformat() if task.created_at else None,
            end_date=task.end_date.isoformat() if task.end_date else None,
            error_msg=task.error_msg,
            trained_model_id=str(task.trained_model_id) if task.trained_model_id else None,
        )
