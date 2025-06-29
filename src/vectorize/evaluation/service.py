"""Service layer for model evaluation."""

import json
from pathlib import Path
from typing import Any

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

from .schemas import EvaluationRequest, EvaluationResponse
from .utils import EvaluationDatasetResolver, resolve_model_path
from .utils.evaluation_engine import EvaluationEngine

__all__ = ["evaluate_model_task_svc", "resolve_evaluation_dataset_svc"]


async def evaluate_model_task_svc(
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

    dataset_path = await EvaluationDatasetResolver.resolve_evaluation_dataset(
        db, evaluation_request
    )

    engine = EvaluationEngine(model_path)

    if evaluation_request.baseline_model_tag:
        baseline_model = await get_ai_model_db(
            db, evaluation_request.baseline_model_tag
        )
        if not baseline_model:
            raise ModelNotFoundError(evaluation_request.baseline_model_tag)

        baseline_model_path = resolve_model_path(baseline_model.model_tag)

        return engine.run_comparison_evaluation(
            evaluation_request, dataset_path, baseline_model_path
        )

    return engine.run_simple_evaluation(evaluation_request, dataset_path)


async def resolve_evaluation_dataset_svc(
    db: AsyncSession, evaluation_request: EvaluationRequest
) -> Path:
    """Resolve the dataset path for evaluation - wrapper for backward compatibility."""
    return await EvaluationDatasetResolver.resolve_evaluation_dataset(
        db, evaluation_request
    )


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
