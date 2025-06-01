"""Schemas für das DPO-Training (Hugging Face TRL)."""


from pydantic import BaseModel, Field

from vectorize.training.models import TrainingTask


class TrainConfig(BaseModel):
    """Hyperparameter für DPO-Training."""

    epochs: int = Field(1, description="Anzahl Trainingsepochen", gt=0)
    learning_rate: float = Field(5e-5, description="Lernrate", gt=0)
    per_device_train_batch_size: int = Field(8, description="Batchgröße pro Gerät", gt=0)


class TrainingStatusResponse(BaseModel):
    """Response-Schema für Trainingsstatus."""

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
    """Request für DPO-Training: erwartet Datensätze im prompt/chosen/rejected-Format (JSONL)."""

    model_id: str = Field(description="ID des lokalen Modells in der Datenbank")
    dataset_ids: list[str] = Field(
        description="IDs der Trainings-Datasets (JSONL, prompt/chosen/rejected)",
        min_length=1,
    )
    output_dir: str = Field(description="Pfad zum Speichern des trainierten Modells")
    epochs: int = Field(1, description="Anzahl Trainingsepochen", gt=0)
    learning_rate: float = Field(5e-5, description="Lernrate", gt=0)
    per_device_train_batch_size: int = Field(
        8, description="Batchgröße pro Gerät", gt=0
    )
