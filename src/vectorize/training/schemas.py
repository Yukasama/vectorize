"""Schemas for DPO training (Hugging Face TRL)."""


from pydantic import BaseModel, Field

from vectorize.training.models import TrainingTask


class TrainConfig(BaseModel):
    """Hyperparameters for DPO training."""

    epochs: int = Field(1, description="Number of training epochs", gt=0)
    learning_rate: float = Field(5e-5, description="Learning rate", gt=0)
    per_device_train_batch_size: int = Field(
        8, description="Batch size per device", gt=0
    )


class TrainingStatusResponse(BaseModel):
    """Response schema for training status."""

    task_id: str
    status: str
    created_at: str | None = None
    end_date: str | None = None
    error_msg: str | None = None
    trained_model_id: str | None = None

    @classmethod
    def from_task(cls, task: TrainingTask) -> "TrainingStatusResponse":
        """Create a TrainingStatusResponse from a TrainingTask."""
        return cls(
            task_id=str(task.id),
            status=task.task_status.name,
            created_at=task.created_at.isoformat() if task.created_at else None,
            end_date=task.end_date.isoformat() if task.end_date else None,
            error_msg=task.error_msg,
            trained_model_id=str(task.trained_model_id)
            if task.trained_model_id else None,
        )


class TrainRequest(BaseModel):
    """Request for DPO training: expects datasets in prompt/chosen/rejected (JSONL)."""

    model_id: str = Field(description="ID of the local model in the database")
    dataset_ids: list[str] = Field(
        description="IDs of training datasets (JSONL, prompt/chosen/rejected)",
        min_length=1,
    )
    output_dir: str = Field(description="Path to save the trained model")
    epochs: int = Field(1, description="Number of training epochs", gt=0)
    learning_rate: float = Field(5e-5, description="Learning rate", gt=0)
    per_device_train_batch_size: int = Field(
        8, description="Batch size per device", gt=0
    )
