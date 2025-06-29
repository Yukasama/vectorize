"""Evaluation engine for running model evaluations."""

from pathlib import Path

from loguru import logger

from ..evaluation import EvaluationMetrics, TrainingEvaluator
from ..schemas import EvaluationRequest, EvaluationResponse


class EvaluationEngine:
    """Handles the actual evaluation process for models."""

    def __init__(self, model_path: str) -> None:
        """Initialize the evaluation engine.

        Args:
            model_path: Path to the model to evaluate
        """
        self.model_path = model_path
        self.evaluator = TrainingEvaluator(model_path)

    def run_simple_evaluation(
        self,
        evaluation_request: EvaluationRequest,
        dataset_path: Path,
    ) -> EvaluationResponse:
        """Run simple evaluation without baseline comparison.

        Args:
            evaluation_request: Evaluation configuration
            dataset_path: Path to the dataset file

        Returns:
            Evaluation response with metrics
        """
        logger.debug(
            "Running simple evaluation",
            model_path=self.model_path,
            dataset_path=str(dataset_path),
            max_samples=evaluation_request.max_samples,
        )

        metrics = self.evaluator.evaluate_dataset(
            dataset_path, evaluation_request.max_samples
        )

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

    def run_comparison_evaluation(
        self,
        evaluation_request: EvaluationRequest,
        dataset_path: Path,
        baseline_model_path: str,
    ) -> EvaluationResponse:
        """Run evaluation with baseline model comparison.

        Args:
            evaluation_request: Evaluation configuration
            dataset_path: Path to the dataset file
            baseline_model_path: Path to the baseline model

        Returns:
            Evaluation response with comparison metrics
        """
        logger.debug(
            "Running comparison evaluation",
            model_path=self.model_path,
            baseline_model_path=baseline_model_path,
            dataset_path=str(dataset_path),
            max_samples=evaluation_request.max_samples,
        )

        comparison_results = self.evaluator.compare_models(
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

    def get_simple_metrics_dict(
        self, dataset_path: Path, max_samples: int | None
    ) -> dict:
        """Get evaluation metrics as dictionary for background tasks.

        Args:
            dataset_path: Path to the dataset file
            max_samples: Maximum number of samples to evaluate

        Returns:
            Dictionary containing evaluation metrics
        """
        metrics = self.evaluator.evaluate_dataset(dataset_path, max_samples)
        return metrics.to_dict()

    def get_comparison_metrics_dict(
        self,
        dataset_path: Path,
        baseline_model_path: str,
        max_samples: int | None,
    ) -> tuple[dict, dict]:
        """Get comparison metrics as dictionaries for background tasks.

        Args:
            dataset_path: Path to the dataset file
            baseline_model_path: Path to the baseline model
            max_samples: Maximum number of samples to evaluate

        Returns:
            Tuple of (trained_metrics_dict, baseline_metrics_dict)
        """
        comparison_results = self.evaluator.compare_models(
            baseline_model_path=baseline_model_path,
            dataset_path=dataset_path,
            max_samples=max_samples,
        )

        trained_metrics = comparison_results["trained"]
        baseline_metrics = comparison_results["baseline"]

        return trained_metrics.to_dict(), baseline_metrics.to_baseline_dict()

    @staticmethod
    def calculate_improvement_summary(
        trained_metrics_dict: dict, baseline_metrics_dict: dict
    ) -> str:
        """Calculate improvement summary from metrics dictionaries.

        Args:
            trained_metrics_dict: Trained model metrics
            baseline_metrics_dict: Baseline model metrics

        Returns:
            Improvement summary string
        """
        trained_metrics = EvaluationMetrics(
            avg_positive_similarity=trained_metrics_dict["avg_positive_similarity"],
            avg_negative_similarity=trained_metrics_dict["avg_negative_similarity"],
            similarity_ratio=trained_metrics_dict["similarity_ratio"],
            spearman_correlation=trained_metrics_dict.get("spearman_correlation", 0.0),
            num_samples=trained_metrics_dict.get(
                "num_samples", trained_metrics_dict.get("total_samples", 0)
            ),
        )

        baseline_metrics = EvaluationMetrics(
            avg_positive_similarity=baseline_metrics_dict["avg_positive_similarity"],
            avg_negative_similarity=baseline_metrics_dict["avg_negative_similarity"],
            similarity_ratio=baseline_metrics_dict["similarity_ratio"],
            spearman_correlation=baseline_metrics_dict.get("spearman_correlation", 0.0),
            num_samples=baseline_metrics_dict.get(
                "num_samples", baseline_metrics_dict.get("total_samples", 0)
            ),
        )

        improvement = trained_metrics.get_improvement_over_baseline(baseline_metrics)

        return (
            f"Similarity ratio improved by "
            f"{improvement['ratio_improvement']:.3f} "
            f"({baseline_metrics.similarity_ratio:.3f} → "
            f"{trained_metrics.similarity_ratio:.3f})"
        )

    @staticmethod
    def calculate_simple_summary(metrics_dict: dict) -> str:
        """Calculate summary from simple metrics dictionary.

        Args:
            metrics_dict: Evaluation metrics dictionary

        Returns:
            Summary string
        """
        return (
            f"Positive similarity: {metrics_dict['avg_positive_similarity']:.3f}, "
            f"Negative similarity: {metrics_dict['avg_negative_similarity']:.3f}, "
            f"Ratio: {metrics_dict['similarity_ratio']:.3f}"
        )
