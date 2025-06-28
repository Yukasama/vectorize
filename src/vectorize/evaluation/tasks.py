"""Evaluation tasks using Dramatiq for background processing."""

import json
from pathlib import Path
from uuid import UUID

import dramatiq
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.ai_model.exceptions import ModelNotFoundError
from vectorize.ai_model.repository import get_ai_model_db
from vectorize.config.db import engine
from vectorize.dataset.repository import get_dataset_db
from vectorize.task.task_status import TaskStatus
from vectorize.training.exceptions import (
    InvalidDatasetIdError,
    TrainingDatasetNotFoundError,
)
from vectorize.training.repository import get_train_task_by_id

from .evaluation import TrainingEvaluator
from .repository import (
    update_evaluation_task_metadata,
    update_evaluation_task_results,
    update_evaluation_task_status,
)
from .schemas import EvaluationRequest
from .utils import resolve_model_path

__all__ = ["run_evaluation_bg"]


def _validate_evaluation_request(evaluation_request: EvaluationRequest) -> None:
    """Validate that evaluation request has correct parameter combination."""
    if evaluation_request.dataset_id and evaluation_request.training_task_id:
        raise ValueError(
            "Cannot specify both dataset_id and training_task_id. "
            "Use dataset_id for explicit dataset or training_task_id "
            "to use the validation dataset from that training."
        )

    if not evaluation_request.dataset_id and not evaluation_request.training_task_id:
        raise ValueError("Must specify either dataset_id or training_task_id.")


async def _resolve_dataset_by_id(db: AsyncSession, dataset_id: str) -> Path:
    """Resolve dataset path using dataset_id."""
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

    return dataset_path


async def _resolve_dataset_by_training_task(
    db: AsyncSession, training_task_id: str
) -> Path:
    """Resolve dataset path using training_task_id."""
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

    return dataset_path


async def resolve_evaluation_dataset(
    db: AsyncSession, evaluation_request: EvaluationRequest
) -> Path:
    """Resolve the dataset path for evaluation based on request parameters."""
    _validate_evaluation_request(evaluation_request)

    if evaluation_request.dataset_id:
        return await _resolve_dataset_by_id(db, evaluation_request.dataset_id)

    if evaluation_request.training_task_id:
        return await _resolve_dataset_by_training_task(
            db, evaluation_request.training_task_id
        )

    raise ValueError("Either dataset_id or training_task_id must be provided")


async def _get_dataset_info(
    db: AsyncSession, evaluation_request: EvaluationRequest
) -> str | None:
    """Get dataset info string for display purposes."""
    if evaluation_request.dataset_id:
        dataset = await get_dataset_db(db, UUID(evaluation_request.dataset_id))
        if dataset:
            return f"Dataset: {dataset.file_name}"
        return f"Dataset ID: {evaluation_request.dataset_id}"
    if evaluation_request.training_task_id:
        return f"Training validation set: {evaluation_request.training_task_id}"
    return None


async def _setup_evaluation_task(
    db: AsyncSession,
    task_uuid: UUID,
    evaluation_request: EvaluationRequest,
) -> tuple[Path, str]:
    """Set up evaluation task and return dataset path and model path."""
    await update_evaluation_task_status(db, task_uuid, TaskStatus.RUNNING, progress=0.1)

    model = await get_ai_model_db(db, evaluation_request.model_tag)
    if not model:
        raise ModelNotFoundError(evaluation_request.model_tag)

    model_path = resolve_model_path(model.model_tag)

    await update_evaluation_task_status(db, task_uuid, TaskStatus.RUNNING, progress=0.2)

    dataset_path = await resolve_evaluation_dataset(db, evaluation_request)
    dataset_info = await _get_dataset_info(db, evaluation_request)

    await update_evaluation_task_metadata(
        db,
        task_uuid,
        model_tag=evaluation_request.model_tag,
        dataset_info=dataset_info,
        baseline_model_tag=evaluation_request.baseline_model_tag,
    )

    await update_evaluation_task_status(db, task_uuid, TaskStatus.RUNNING, progress=0.3)

    return dataset_path, model_path


async def _run_baseline_evaluation(
    db: AsyncSession,
    task_uuid: UUID,
    evaluator: TrainingEvaluator,
    config: dict,
) -> None:
    """Run evaluation with baseline comparison."""
    baseline_model_tag = config["baseline_model_tag"]
    dataset_path = config["dataset_path"]
    max_samples = config["max_samples"]

    baseline_model = await get_ai_model_db(db, baseline_model_tag)
    if not baseline_model:
        raise ModelNotFoundError(baseline_model_tag)

    baseline_model_path = resolve_model_path(baseline_model.model_tag)

    await update_evaluation_task_status(db, task_uuid, TaskStatus.RUNNING, progress=0.5)

    comparison_results = evaluator.compare_models(
        baseline_model_path=baseline_model_path,
        dataset_path=dataset_path,
        max_samples=max_samples,
    )
    trained_metrics = comparison_results["trained"]
    baseline_metrics = comparison_results["baseline"]

    improvement = trained_metrics.get_improvement_over_baseline(baseline_metrics)
    summary = (
        f"Similarity ratio improved by "
        f"{improvement['ratio_improvement']:.3f} "
        f"({baseline_metrics.similarity_ratio:.3f} → "
        f"{trained_metrics.similarity_ratio:.3f})"
    )

    await update_evaluation_task_status(db, task_uuid, TaskStatus.RUNNING, progress=0.9)

    await update_evaluation_task_results(
        db,
        task_uuid,
        evaluation_metrics=json.dumps(trained_metrics.to_dict()),
        baseline_metrics=json.dumps(baseline_metrics.to_baseline_dict()),
        evaluation_summary=summary,
    )


async def _run_simple_evaluation(
    db: AsyncSession,
    task_uuid: UUID,
    evaluator: TrainingEvaluator,
    config: dict,
) -> None:
    """Run simple evaluation without baseline."""
    dataset_path = config["dataset_path"]
    max_samples = config["max_samples"]

    await update_evaluation_task_status(db, task_uuid, TaskStatus.RUNNING, progress=0.5)

    metrics = evaluator.evaluate_dataset(dataset_path, max_samples)

    summary = (
        f"Positive similarity: {metrics.avg_positive_similarity:.3f}, "
        f"Negative similarity: {metrics.avg_negative_similarity:.3f}, "
        f"Ratio: {metrics.similarity_ratio:.3f}"
    )

    await update_evaluation_task_status(db, task_uuid, TaskStatus.RUNNING, progress=0.9)

    await update_evaluation_task_results(
        db,
        task_uuid,
        evaluation_metrics=json.dumps(metrics.to_dict()),
        evaluation_summary=summary,
    )


@dramatiq.actor(max_retries=3)
async def run_evaluation_bg(
    evaluation_request_dict: dict,
    task_id: str,
) -> None:
    """Run model evaluation in the background using Dramatiq.

    Args:
        evaluation_request_dict: Evaluation configuration as dict (JSON serializable)
        task_id: Evaluation task ID as string
    """
    async with AsyncSession(engine, expire_on_commit=False) as db:
        try:
            evaluation_request = EvaluationRequest.model_validate(
                evaluation_request_dict
            )
            task_uuid = UUID(task_id)

            logger.debug(
                "Starting background evaluation task",
                task_id=task_id,
                model_tag=evaluation_request.model_tag,
                dataset_id=evaluation_request.dataset_id,
                training_task_id=evaluation_request.training_task_id,
                baseline_model_tag=evaluation_request.baseline_model_tag,
                max_samples=evaluation_request.max_samples,
            )

            dataset_path, model_path = await _setup_evaluation_task(
                db, task_uuid, evaluation_request
            )

            evaluator = TrainingEvaluator(model_path)

            if evaluation_request.baseline_model_tag:
                await _run_baseline_evaluation(
                    db,
                    task_uuid,
                    evaluator,
                    {
                        "baseline_model_tag": evaluation_request.baseline_model_tag,
                        "dataset_path": dataset_path,
                        "max_samples": evaluation_request.max_samples,
                    },
                )
            else:
                await _run_simple_evaluation(
                    db,
                    task_uuid,
                    evaluator,
                    {
                        "dataset_path": dataset_path,
                        "max_samples": evaluation_request.max_samples,
                    },
                )

            await update_evaluation_task_status(
                db, task_uuid, TaskStatus.DONE, progress=1.0
            )

            logger.info(
                "Evaluation task completed successfully",
                task_id=task_id,
                model_tag=evaluation_request.model_tag,
            )

        except Exception as e:
            logger.error(
                "Error in evaluation background task",
                task_id=task_id,
                model_tag=evaluation_request_dict.get("model_tag", "unknown"),
                error=str(e),
                exc_info=True,
            )
            try:
                await update_evaluation_task_status(
                    db, UUID(task_id), TaskStatus.FAILED, progress=0.0
                )
            except Exception:
                logger.error("Failed to update task status to FAILED", task_id=task_id)
            raise
