"""Evaluation utilities for SBERT training quality assessment.

Computes cosine similarity metrics between question-positive-negative triplets
to assess training effectiveness. Main metrics:
- Average cosine similarity between question and positive examples
- Average cosine similarity between question and negative examples
- Ratio of positive to negative similarities (should be > 1)
- Spearman correlation for similarity ranking
"""

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from .utils import DatasetValidator, SimilarityCalculator

__all__ = ["EvaluationMetrics", "TrainingEvaluator"]

TRAINING_SUCCESS_THRESHOLD = 1.2


class EvaluationMetrics:
    """Container for training evaluation metrics."""

    def __init__(
        self,
        avg_positive_similarity: float,
        avg_negative_similarity: float,
        similarity_ratio: float,
        spearman_correlation: float,
        num_samples: int,
    ) -> None:
        """Initialize evaluation metrics.

        Args:
            avg_positive_similarity: Average cosine similarity (question, positive)
            avg_negative_similarity: Average cosine similarity (question, negative)
            similarity_ratio: Ratio of positive to negative similarities
            spearman_correlation: Spearman correlation coefficient
            num_samples: Number of triplets evaluated
        """
        self.avg_positive_similarity = avg_positive_similarity
        self.avg_negative_similarity = avg_negative_similarity
        self.similarity_ratio = similarity_ratio
        self.spearman_correlation = spearman_correlation
        self.num_samples = num_samples

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "avg_positive_similarity": self.avg_positive_similarity,
            "avg_negative_similarity": self.avg_negative_similarity,
            "similarity_ratio": self.similarity_ratio,
            "spearman_correlation": self.spearman_correlation,
            "num_samples": self.num_samples,
            "is_training_successful": self.is_training_successful(),
        }

    def to_baseline_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary format for baseline models."""
        return {
            "avg_positive_similarity": self.avg_positive_similarity,
            "avg_negative_similarity": self.avg_negative_similarity,
            "similarity_ratio": self.similarity_ratio,
            "spearman_correlation": self.spearman_correlation,
            "num_samples": self.num_samples,
        }

    def is_training_successful(self) -> bool:
        """Simple heuristic to determine if training was successful.

        Returns:
            True if positive similarities > negative similarities and ratio > 1.2
        """
        return (
            self.avg_positive_similarity > self.avg_negative_similarity
            and self.similarity_ratio > TRAINING_SUCCESS_THRESHOLD
        )

    def __str__(self) -> str:
        """String representation of metrics."""
        return (
            f"EvaluationMetrics(\n"
            f"  avg_positive_similarity={self.avg_positive_similarity:.4f}\n"
            f"  avg_negative_similarity={self.avg_negative_similarity:.4f}\n"
            f"  similarity_ratio={self.similarity_ratio:.4f}\n"
            f"  spearman_correlation={self.spearman_correlation:.4f}\n"
            f"  num_samples={self.num_samples}\n"
            f"  training_successful={self.is_training_successful()}\n"
            f")"
        )

    def baseline_str(self) -> str:
        """String representation for baseline metrics (without training_successful)."""
        return (
            f"BaselineMetrics(\n"
            f"  avg_positive_similarity={self.avg_positive_similarity:.4f}\n"
            f"  avg_negative_similarity={self.avg_negative_similarity:.4f}\n"
            f"  similarity_ratio={self.similarity_ratio:.4f}\n"
            f"  spearman_correlation={self.spearman_correlation:.4f}\n"
            f"  num_samples={self.num_samples}\n"
            f")"
        )


class TrainingEvaluator:
    """Evaluates SBERT training quality using cosine similarity metrics."""

    def __init__(self, model_path: str) -> None:
        """Initialize evaluator with trained model.

        Args:
            model_path: Path to the trained SentenceTransformer model
        """
        self.model_path = model_path
        self.model: SentenceTransformer | None = None

    def _load_model(self) -> SentenceTransformer:
        """Load the SentenceTransformer model."""
        if self.model is None:
            logger.debug("Loading model for evaluation", model_path=self.model_path)
            self.model = SentenceTransformer(self.model_path)
        return self.model

    def evaluate_dataset(
        self, dataset_path: Path, max_samples: int | None = None
    ) -> EvaluationMetrics:
        """Evaluate training quality on a dataset.

        Args:
            dataset_path: Path to evaluation dataset (JSONL)
            max_samples: Optional limit on number of samples to evaluate

        Returns:
            EvaluationMetrics with computed similarity metrics

        Raises:
            DatasetValidationError: If dataset is invalid
        """
        logger.info("Starting evaluation", dataset_path=str(dataset_path))

        df = DatasetValidator.validate_dataset(dataset_path)

        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)
            logger.debug(f"Limited evaluation to {max_samples} samples")

        model = self._load_model()

        questions = df["Question"].tolist()
        positives = df["Positive"].tolist()
        negatives = df["Negative"].tolist()

        logger.debug(f"Computing embeddings for {len(questions)} triplets")

        positive_similarities, negative_similarities = (
            SimilarityCalculator.compute_cosine_similarities(
                model, questions, positives, negatives
            )
        )

        pos_sims = np.array(positive_similarities)
        neg_sims = np.array(negative_similarities)

        avg_positive_similarity = float(np.mean(pos_sims))
        avg_negative_similarity = float(np.mean(neg_sims))

        similarity_ratio = SimilarityCalculator.compute_similarity_ratio(
            avg_positive_similarity, avg_negative_similarity
        )

        spearman_corr = SimilarityCalculator.compute_spearman_correlation(
            positive_similarities, negative_similarities
        )

        metrics = EvaluationMetrics(
            avg_positive_similarity=avg_positive_similarity,
            avg_negative_similarity=avg_negative_similarity,
            similarity_ratio=similarity_ratio,
            spearman_correlation=spearman_corr,
            num_samples=len(questions),
        )

        logger.info("Evaluation completed", metrics=str(metrics))
        return metrics

    def compare_models(
        self,
        baseline_model_path: str,
        dataset_path: Path,
        max_samples: int | None = None,
    ) -> dict[str, EvaluationMetrics]:
        """Compare trained model against baseline model.

        Args:
            baseline_model_path: Path to baseline model for comparison
            dataset_path: Path to evaluation dataset
            max_samples: Optional limit on samples

        Returns:
            Dictionary with 'trained' and 'baseline' metrics
        """
        logger.debug(
            "Comparing models", trained=self.model_path, baseline=baseline_model_path
        )

        trained_metrics = self.evaluate_dataset(dataset_path, max_samples)

        baseline_evaluator = TrainingEvaluator(baseline_model_path)
        baseline_metrics = baseline_evaluator.evaluate_dataset(
            dataset_path, max_samples
        )

        # Log baseline metrics without training_successful
        logger.info(
            "Baseline evaluation completed",
            metrics=baseline_metrics.baseline_str(),
        )

        return {"trained": trained_metrics, "baseline": baseline_metrics}
