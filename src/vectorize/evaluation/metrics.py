"""Evaluation metrics data models."""

from typing import Any


class EvaluationMetrics:
    """Container for training evaluation metrics."""

    def __init__(
        self,
        avg_positive_similarity: float,
        avg_negative_similarity: float,
        similarity_ratio: float,
        total_samples: int,
    ) -> None:
        """Initialize evaluation metrics.

        Args:
            avg_positive_similarity: Average similarity for positive pairs
            avg_negative_similarity: Average similarity for negative pairs
            similarity_ratio: Ratio of positive to negative similarity
            total_samples: Total number of samples evaluated
        """
        self.avg_positive_similarity = avg_positive_similarity
        self.avg_negative_similarity = avg_negative_similarity
        self.similarity_ratio = similarity_ratio
        self.total_samples = total_samples

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary format.

        Returns:
            Dictionary containing all metrics
        """
        return {
            "avg_positive_similarity": self.avg_positive_similarity,
            "avg_negative_similarity": self.avg_negative_similarity,
            "similarity_ratio": self.similarity_ratio,
            "total_samples": self.total_samples,
        }

    def to_baseline_dict(self) -> dict[str, Any]:
        """Convert metrics to baseline dictionary format.

        Returns:
            Dictionary containing baseline metrics with 'baseline_' prefix
        """
        return {
            "baseline_avg_positive_similarity": self.avg_positive_similarity,
            "baseline_avg_negative_similarity": self.avg_negative_similarity,
            "baseline_similarity_ratio": self.similarity_ratio,
            "baseline_total_samples": self.total_samples,
        }

    def get_improvement_over_baseline(
            self, baseline: "EvaluationMetrics") -> dict[str, float]:
        """Calculate improvement over baseline metrics.

        Args:
            baseline: Baseline metrics to compare against

        Returns:
            Dictionary containing improvement calculations
        """
        return {
            "ratio_improvement": self.similarity_ratio - baseline.similarity_ratio,
            "positive_improvement": (
                self.avg_positive_similarity - baseline.avg_positive_similarity
            ),
            "negative_improvement": (
                self.avg_negative_similarity - baseline.avg_negative_similarity
            ),
        }

    def __str__(self) -> str:
        """String representation of metrics."""
        return (
            f"EvaluationMetrics("
            f"avg_positive={self.avg_positive_similarity:.3f}, "
            f"avg_negative={self.avg_negative_similarity:.3f}, "
            f"ratio={self.similarity_ratio:.3f}, "
            f"samples={self.total_samples})"
        )
