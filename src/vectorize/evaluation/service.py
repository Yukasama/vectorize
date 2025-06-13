"""Service layer for model evaluation."""

from pathlib import Path
from uuid import UUID

from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.ai_model.exceptions import ModelNotFoundError
from vectorize.ai_model.repository import get_ai_model_db
from vectorize.dataset.repository import get_dataset_db
from vectorize.training.exceptions import (
    InvalidDatasetIdError,
    TrainingDatasetNotFoundError,
)
from vectorize.training.utils.safetensors_finder import find_safetensors_file
from vectorize.training.utils.uuid_validator import is_valid_uuid

from .evaluation import TrainingEvaluator
from .schemas import EvaluationRequest, EvaluationResponse

__all__ = ["evaluate_model_task"]


def resolve_model_path(model_tag: str) -> str:
    """Resolve model tag to actual model path.

    Uses recursive search to find the correct model directory structure.
    Works with both HuggingFace cache format and direct model directories.

    Args:
        model_tag: Model tag/path

    Returns:
        str: Resolved model path containing model files

    Raises:
        FileNotFoundError: If model path cannot be resolved
    """
    base_path = Path("data/models") / model_tag

    if not base_path.exists():
        raise FileNotFoundError(f"Model directory not found: {base_path}")

    snapshots_dir = base_path / "snapshots"
    if snapshots_dir.exists():
        snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
        if snapshot_dirs:
            candidate_path = snapshot_dirs[0]
            if (candidate_path / "config.json").exists():
                return str(candidate_path)
            safetensors_path = find_safetensors_file(str(candidate_path))
            if safetensors_path:
                return str(candidate_path)

        raise FileNotFoundError(f"No valid model found in snapshots: {snapshots_dir}")

    if (base_path / "config.json").exists():
        return str(base_path)

    safetensors_path = find_safetensors_file(str(base_path))
    if safetensors_path:
        return str(Path(safetensors_path).parent)

    raise FileNotFoundError(f"No valid model files found in: {base_path}")


async def evaluate_model_task(
    db: AsyncSession,
    evaluation_request: EvaluationRequest,
) -> EvaluationResponse:
    """Service entry point for model evaluation.

    Args:
        db (AsyncSession): Database session.
        evaluation_request (EvaluationRequest): Evaluation configuration.

    Returns:
        EvaluationResponse: Evaluation results.

    Raises:
        InvalidDatasetIdError: If dataset ID is invalid.
        TrainingDatasetNotFoundError: If dataset file not found.
        ModelNotFoundError: If model tag not found.
    """
    model = await get_ai_model_db(db, evaluation_request.model_tag)
    if not model:
        raise ModelNotFoundError(evaluation_request.model_tag)

    model_path = resolve_model_path(model.model_tag)

    if not is_valid_uuid(evaluation_request.dataset_id):
        raise InvalidDatasetIdError(evaluation_request.dataset_id)

    dataset_uuid = UUID(evaluation_request.dataset_id)
    dataset = await get_dataset_db(db, dataset_uuid)
    if not dataset:
        raise TrainingDatasetNotFoundError(
            f"Dataset not found: {evaluation_request.dataset_id}"
        )

    dataset_path = Path("data/datasets") / dataset.file_name
    if not dataset_path.exists():
        raise TrainingDatasetNotFoundError(f"Dataset file not found: {dataset_path}")

    evaluator = TrainingEvaluator(model_path)

    if evaluation_request.baseline_model_tag:
        baseline_model = await get_ai_model_db(
            db, evaluation_request.baseline_model_tag
        )
        if not baseline_model:
            raise ModelNotFoundError(evaluation_request.baseline_model_tag)

        baseline_model_path = resolve_model_path(baseline_model.model_tag)

        comparison_results = evaluator.compare_models(
            baseline_model_path=baseline_model_path,
            dataset_path=dataset_path,
            max_samples=evaluation_request.max_samples,
        )
        trained_metrics = comparison_results["trained"]
        baseline_metrics = comparison_results["baseline"]

        improvement = (
            trained_metrics.similarity_ratio - baseline_metrics.similarity_ratio
        )
        success_status = (
            "successful" if trained_metrics.is_training_successful() else "unsuccessful"
        )
        summary = (
            f"Training {success_status}. "
            f"Similarity ratio improved by {improvement:.3f} "
            f"({baseline_metrics.similarity_ratio:.3f} → "
            f"{trained_metrics.similarity_ratio:.3f})"
        )

        return EvaluationResponse(
            model_tag=evaluation_request.model_tag,
            dataset_used=str(dataset_path),
            metrics=trained_metrics.to_dict(),
            baseline_metrics=baseline_metrics.to_dict(),
            evaluation_summary=summary,
            training_successful=trained_metrics.is_training_successful(),
        )
    metrics = evaluator.evaluate_dataset(dataset_path, evaluation_request.max_samples)

    success_status = (
        "successful" if metrics.is_training_successful() else "unsuccessful"
    )
    summary = (
        f"Training {success_status}. "
        f"Positive similarity: {metrics.avg_positive_similarity:.3f}, "
        f"Negative similarity: {metrics.avg_negative_similarity:.3f}, "
        f"Ratio: {metrics.similarity_ratio:.3f}"
    )

    return EvaluationResponse(
        model_tag=evaluation_request.model_tag,
        dataset_used=str(dataset_path),
        metrics=metrics.to_dict(),
        evaluation_summary=summary,
        training_successful=metrics.is_training_successful(),
    )
