"""Service layer for model evaluation."""

import json
from pathlib import Path
from typing import Any
from uuid import UUID

import torch
from loguru import logger
from mteb import MTEB, get_tasks
from mteb.abstasks import AbsTask
from sentence_transformers import SentenceTransformer
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.ai_model.exceptions import ModelNotFoundError
from vectorize.ai_model.model_source import ModelSource
from vectorize.ai_model.models import AIModel
from vectorize.ai_model.repository import get_ai_model_db
from vectorize.dataset.repository import get_dataset_db
from vectorize.training.exceptions import (
    InvalidDatasetIdError,
    TrainingDatasetNotFoundError,
)
from vectorize.training.repository import get_train_task_by_id

from .evaluation import TrainingEvaluator
from .schemas import EvaluationRequest, EvaluationResponse
from .utils import resolve_model_path

__all__ = ["evaluate_model_task", "resolve_evaluation_dataset"]


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
        ValueError: If neither dataset_id nor training_task_id is provided.
        InvalidDatasetIdError: If dataset ID is invalid.
        TrainingDatasetNotFoundError: If dataset file not found.
        ModelNotFoundError: If model tag not found.
    """
    model = await get_ai_model_db(db, evaluation_request.model_tag)
    if not model:
        raise ModelNotFoundError(evaluation_request.model_tag)

    model_path = resolve_model_path(model.model_tag)

    # Resolve dataset using either dataset_id or training_task_id
    dataset_path = await resolve_evaluation_dataset(db, evaluation_request)

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

        improvement = trained_metrics.get_improvement_over_baseline(baseline_metrics)
        summary = (
            f"Similarity ratio improved by {improvement['ratio_improvement']:.3f} "
            f"({baseline_metrics.similarity_ratio:.3f} → "
            f"{trained_metrics.similarity_ratio:.3f})"
        )

        return EvaluationResponse(
            model_tag=evaluation_request.model_tag,
            dataset_used=str(dataset_path),
            metrics=trained_metrics.to_dict(),
            baseline_metrics=baseline_metrics.to_baseline_dict(),
            evaluation_summary=summary,
        )

    metrics = evaluator.evaluate_dataset(dataset_path, evaluation_request.max_samples)

    summary = (
        f"Positive similarity: {metrics.avg_positive_similarity:.3f}, "
        f"Negative similarity: {metrics.avg_negative_similarity:.3f}, "
        f"Ratio: {metrics.similarity_ratio:.3f}"
    )

    return EvaluationResponse(
        model_tag=evaluation_request.model_tag,
        dataset_used=str(dataset_path),
        metrics=metrics.to_dict(),
        evaluation_summary=summary,
    )


async def resolve_evaluation_dataset(
    db: AsyncSession, evaluation_request: EvaluationRequest
) -> Path:
    """Resolve the dataset path for evaluation.

    Uses either the explicit dataset_id or gets the validation dataset
    from the training_task_id.

    Args:
        db: Database session
        evaluation_request: Evaluation request with dataset_id and/or training_task_id

    Returns:
        Path to the dataset file

    Raises:
        ValueError: If neither dataset_id nor training_task_id is provided,
                   or if both are provided.
        InvalidDatasetIdError: If dataset ID is invalid.
        TrainingDatasetNotFoundError: If dataset or training task not found.
    """
    _validate_dataset_resolution_input(evaluation_request)

    if evaluation_request.dataset_id:
        return await _resolve_explicit_dataset(db, evaluation_request.dataset_id)

    if evaluation_request.training_task_id is None:
        raise ValueError("training_task_id must be set at this point")
    return await _resolve_training_validation_dataset(
        db, evaluation_request.training_task_id
    )


def _validate_dataset_resolution_input(evaluation_request: EvaluationRequest) -> None:
    """Validate dataset resolution input parameters."""
    if evaluation_request.dataset_id and evaluation_request.training_task_id:
        raise ValueError(
            "Cannot specify both dataset_id and training_task_id. "
            "Use dataset_id for explicit dataset or training_task_id "
            "to use the validation dataset from that training."
        )

    if not evaluation_request.dataset_id and not evaluation_request.training_task_id:
        raise ValueError("Must specify either dataset_id or training_task_id.")


async def _resolve_explicit_dataset(db: AsyncSession, dataset_id: str) -> Path:
    """Resolve explicit dataset by ID."""
    try:
        dataset_uuid = UUID(dataset_id)
    except ValueError as exc:
        raise InvalidDatasetIdError(dataset_id) from exc

    dataset = await get_dataset_db(db, dataset_uuid)
    if not dataset:
        raise TrainingDatasetNotFoundError(f"Dataset not found: {dataset_id}")

    dataset_path = Path("data/datasets") / dataset.file_name
    if not dataset_path.exists():
        raise TrainingDatasetNotFoundError(f"Dataset file not found: {dataset_path}")

    logger.info(
        "Using explicit dataset for evaluation",
        dataset_id=dataset_id,
        dataset_path=str(dataset_path)
    )
    return dataset_path


async def _resolve_training_validation_dataset(
    db: AsyncSession, training_task_id: str
) -> Path:
    """Resolve validation dataset from training task."""
    try:
        task_uuid = UUID(training_task_id)
    except ValueError as exc:
        raise InvalidDatasetIdError(training_task_id) from exc

    training_task = await get_train_task_by_id(db, task_uuid)
    if not training_task:
        raise TrainingDatasetNotFoundError(
            f"Training task not found: {training_task_id}"
        )

    if not training_task.validation_dataset_path:
        raise TrainingDatasetNotFoundError(
            f"Training task {training_task_id} has no validation dataset path"
        )

    dataset_path = Path(training_task.validation_dataset_path)
    if not dataset_path.exists():
        raise TrainingDatasetNotFoundError(
            f"Validation dataset file not found: {dataset_path}"
        )

    logger.info(
        "Using validation dataset from training task",
        training_task_id=training_task_id,
        validation_dataset_path=str(dataset_path)
    )
    return dataset_path


SELECTED_MTEB_TASKS = ["STSBenchmark", "BIOSSES", "SICK-R"]
CACHE_BASE_DIR = "/app/data/models"


class EvaluationService:
    """Handles the full lifecycle of running MTEB benchmark on a cached model."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def run_benchmark(self, model_tag: str) -> dict[str, Any]:
        """Run selected MTEB tasks on a cached model identified by its tag.

        Args:
            model_tag: Unique model tag or UUID.

        Returns:
            Dictionary with benchmark results.
        """
        logger.info("Running MTEB benchmark for model: {}", model_tag)

        model = await get_ai_model_db(self.db, model_tag)
        if not model:
            logger.error("Model with tag {} not found", model_tag)
            raise ModelNotFoundError()

        cache_path = resolve_cache_path(model)
        logger.debug("Resolved cache path: {}", cache_path)

        transformer: SentenceTransformer = self._load_model(cache_path)
        tasks: list[AbsTask] = get_tasks(tasks=SELECTED_MTEB_TASKS)

        benchmark = MTEB(tasks=tasks)
        results = benchmark.run(
            model=transformer,
            output_folder=None,
            eval_splits=["test"],
            eval_subsets=[]
        )

        return self._serialize_results(results)

    @staticmethod
    def _load_model(model_path: Path) -> SentenceTransformer:
        """Load a SentenceTransformer model and move to CUDA if available.

        Args:
            model_path: Filesystem path to the cached model.

        Returns:
            Loaded and prepared SentenceTransformer.
        """
        logger.debug("Loading model from: {}", model_path)
        model = SentenceTransformer(str(model_path))

        if torch.cuda.is_available():
            model = model.to("cuda")
            logger.debug("Model loaded on CUDA: {}", torch.cuda.get_device_name(0))
        else:
            logger.warning("CUDA not available. Using CPU.")

        return model

    @staticmethod
    def _serialize_results(results: list[Any]) -> dict[str, Any]:
        """Serialize benchmark results to JSON-serializable dict.

        Args:
            results: List of TaskResult-like objects.

        Returns:
            JSON-compatible dictionary of results.
        """
        def serializer(obj: object) -> dict[str, Any]:
            if obj.__class__.__name__ == "TaskResult":
                return obj.__dict__
            raise TypeError("Unsupported type {} for serialization",
                obj.__class__.__name__)

        return json.loads(json.dumps(results, default=serializer))


def resolve_cache_path(model: AIModel) -> Path:
    """Resolve the full local snapshot path for a cached model.

    Args:
        model: AIModel instance

    Returns:
        Path to model snapshot directory.

    Raises:
        ValueError, FileNotFoundError, RuntimeError
    """
    source_map = {
        ModelSource.HUGGINGFACE: "hf_home",
        ModelSource.GITHUB: "gh_home",
        ModelSource.LOCAL: "local_cache",
    }

    source_dir = source_map.get(model.source)
    if not source_dir:
        raise ValueError(f"Unsupported model source: {model.source}")

    base_path = Path(CACHE_BASE_DIR) / f"models--{model.name.replace('/', '--')}"
    snapshot_dir = base_path / "snapshots"

    if not snapshot_dir.exists():
        raise FileNotFoundError("Snapshot folder not found at: {}", snapshot_dir)

    snapshots = [p for p in snapshot_dir.iterdir() if p.is_dir()]
    if not snapshots:
        raise FileNotFoundError("No snapshot folders found in: {}", snapshot_dir)
    if len(snapshots) > 1:
        raise RuntimeError("Multiple snapshot folders found in: {}", snapshot_dir)

    return snapshots[0]
