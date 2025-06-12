"""Schemas for model evaluation endpoints."""

from pydantic import BaseModel, Field

__all__ = ["EvaluationRequest", "EvaluationResponse"]


class EvaluationRequest(BaseModel):
    """Request for evaluating a trained model."""

    model_tag: str = Field(
        description="Tag of the trained model from the database"
    )
    dataset_id: str = Field(
        description="ID of the dataset to use for evaluation"
    )
    max_samples: int | None = Field(
        default=1000,
        description="Maximum number of samples to evaluate (default: 1000)",
        gt=0
    )
    baseline_model_tag: str | None = Field(
        default=None,
        description="Optional tag of baseline model for comparison"
    )


class EvaluationResponse(BaseModel):
    """Response for model evaluation."""

    model_tag: str
    dataset_used: str
    metrics: dict
    baseline_metrics: dict | None = None
    evaluation_summary: str
    training_successful: bool = Field(
        description="Whether the training was deemed successful based on metrics"
    )
