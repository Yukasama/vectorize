"""Schemas for the training API."""


from pydantic import BaseModel, Field, field_validator

from vectorize.training.models import TrainingTask


class TrainConfig(BaseModel):
    """Hyperparameters for model training."""

    epochs: int = Field(1, description="Number of training epochs", gt=0)
    learning_rate: float = Field(5e-5, description="Learning rate for training")
    per_device_train_batch_size: int = Field(8, description="Batch size per device")


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


class TrainRequest(BaseModel):
    """Request body for model training."""

    model_id: str = Field(
        description=(
            "ID des lokalen Modells in der Datenbank (Pfad wird im Backend ermittelt)"
        ),
    )
    dataset_paths: list[str] = Field(
        description="Paths to the training datasets (local, multiple allowed)",
        min_length=1
    )
    output_dir: str = Field(description="Path to save the trained model")
    epochs: int = Field(1, description="Number of training epochs", gt=0)
    learning_rate: float = Field(5e-5, description="Learning rate for training", gt=0)
    per_device_train_batch_size: int = Field(8, description="Batch size per device", gt=0)

    @field_validator("dataset_paths")
    @classmethod
    def check_paths(cls, paths: list[str]) -> list[str]:
        """Validate that all dataset paths end with .csv"""
        for p in paths:
            if not p.endswith(".csv"):
                raise ValueError("All dataset paths must be .csv files")
        return paths
