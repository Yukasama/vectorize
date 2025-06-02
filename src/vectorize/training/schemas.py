"""Schemas for DPO training (Hugging Face TRL)."""

from pydantic import BaseModel, Field
from typing import Optional

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
    progress: float | None = None

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
            if task.trained_model_id
            else None,
            progress=task.progress if hasattr(task, "progress") else None,
        )


class TrainRequest(BaseModel):
    """Request for DPO training: expects datasets in prompt/chosen/rejected (JSONL).
    All DPOConfig parameters supported. Important ones are required, others optional.
    """

    model_id: str = Field(description="ID of the local model in the database")
    dataset_ids: list[str] = Field(
        description="IDs of training datasets (JSONL, prompt/chosen/rejected)",
        min_length=1,
    )
    epochs: int = Field(1, description="Number of training epochs", gt=0)
    learning_rate: float = Field(5e-5, description="Learning rate", gt=0)
    per_device_train_batch_size: int = Field(
        8, description="Batch size per device", gt=0
    )
    # Optionale DPOConfig-Parameter (Beispiele, ggf. nach Bedarf erweitern)
    weight_decay: Optional[float] = Field(
        None, description="Weight decay (L2 regularization)"
    )
    warmup_steps: Optional[int] = Field(
        None, description="Number of warmup steps"
    )
    logging_steps: Optional[int] = Field(
        None, description="Log every X steps"
    )
    save_steps: Optional[int] = Field(
        None, description="Save checkpoint every X steps"
    )
    max_grad_norm: Optional[float] = Field(
        None, description="Maximum gradient norm for clipping"
    )
    gradient_accumulation_steps: Optional[int] = Field(
        None, description="Number of steps to accumulate gradients"
    )
    fp16: Optional[bool] = Field(
        None, description="Use mixed precision training (fp16)"
    )
    bf16: Optional[bool] = Field(
        None, description="Use bfloat16 precision"
    )
    # ...weitere DPOConfig-Parameter nach Bedarf ergänzen...
