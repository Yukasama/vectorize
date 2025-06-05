"""Schemas for SBERT/SentenceTransformer triplet training."""


from pydantic import BaseModel, Field
from vectorize.training.models import TrainingTask


class TrainRequest(BaseModel):
    """Request für SBERT Triplet-Training: CSVs mit question,positive,negative. Alle wichtigen und optionalen sentence-transformers Parameter unterstützt."""

    # Pflichtparameter
    model_tag: str = Field(description="Tag des lokalen Modells in der Datenbank")
    dataset_ids: list[str] = Field(
        description="IDs der Trainingsdatensätze (CSV, Spalten: question,positive,negative)",
        min_length=1,
    )
    epochs: int = Field(1, description="Anzahl Trainingsepochen", gt=0)
    per_device_train_batch_size: int = Field(
        8, description="Batch-Größe pro Gerät", gt=0
    )
    learning_rate: float = Field(
        2e-5, description="Learning Rate", gt=0
    )

    # Optionale sentence-transformers Parameter
    warmup_steps: int | None = Field(
        None, description="Anzahl Warmup-Schritte"
    )
    optimizer_name: str | None = Field(
        None, description="Optimizer (z.B. AdamW, Adam, RMSprop)"
    )
    scheduler: str | None = Field(
        None, description="Lernraten-Scheduler (z.B. 'constantlr', 'warmuplinear')"
    )
    weight_decay: float | None = Field(
        None, description="Weight Decay (L2 Regularisierung)"
    )
    max_grad_norm: float | None = Field(
        None, description="Maximaler Gradient-Norm für Clipping"
    )
    use_amp: bool | None = Field(
        None, description="Automatisches Mixed Precision Training (AMP) nutzen"
    )
    show_progress_bar: bool | None = Field(
        None, description="Fortschrittsbalken anzeigen"
    )
    evaluation_steps: int | None = Field(
        None, description="Evaluiere alle X Schritte (optional, falls Val-Set)"
    )
    output_path: str | None = Field(
        None, description="Pfad zum Speichern des Modells (optional, wird sonst automatisch gewählt)"
    )
    save_best_model: bool | None = Field(
        None, description="Bestes Modell speichern (bei Val-Set)"
    )
    save_each_epoch: bool | None = Field(
        None, description="Modell nach jeder Epoche speichern"
    )
    save_optimizer_state: bool | None = Field(
        None, description="Optimizer-Status mit speichern"
    )
    dataloader_num_workers: int | None = Field(
        None, description="Anzahl DataLoader-Worker (default: 0)"
    )
    device: str | None = Field(
        None, description="Gerät für Training ('cpu', 'cuda', 'mps', etc.)"
    )
    # ...weitere optionale Parameter können ergänzt werden


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
