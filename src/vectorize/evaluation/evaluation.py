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
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from ..training.exceptions import DatasetValidationError

__all__ = ["EvaluationMetrics", "TrainingEvaluator"]

# Training success threshold for similarity ratio
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


class TrainingEvaluator:
    """Evaluates SBERT training quality using cosine similarity metrics."""

    REQUIRED_COLUMNS = {"Question", "Positive", "Negative"}

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

    def _validate_dataset(self, dataset_path: Path) -> pd.DataFrame:
        """Validate and load dataset for evaluation.

        Args:
            dataset_path: Path to JSONL dataset file

        Returns:
            Validated DataFrame

        Raises:
            DatasetValidationError: If dataset is invalid
        """
        try:
            df = pd.read_json(dataset_path, lines=True)
        except Exception as exc:
            raise DatasetValidationError(
                f"Invalid JSONL file {dataset_path}: {exc}"
            ) from exc

        missing_cols = self.REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            raise DatasetValidationError(
                f"Missing columns in {dataset_path}: {missing_cols}"
            )

        if df.empty:
            raise DatasetValidationError(f"Dataset {dataset_path} is empty")

        for col in self.REQUIRED_COLUMNS:
            if df[col].isnull().any():
                raise DatasetValidationError(
                    f"Column '{col}' contains null values in {dataset_path}"
                )

        return df

    def evaluate_dataset(  # noqa: PLR0914  # Complex evaluation requires many variables
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

        df = self._validate_dataset(dataset_path)

        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)
            logger.debug(f"Limited evaluation to {max_samples} samples")

        model = self._load_model()

        questions = df["Question"].tolist()
        positives = df["Positive"].tolist()
        negatives = df["Negative"].tolist()

        logger.debug(f"Computing embeddings for {len(questions)} triplets")

        question_embeddings = model.encode(questions, show_progress_bar=False)
        positive_embeddings = model.encode(positives, show_progress_bar=False)
        negative_embeddings = model.encode(negatives, show_progress_bar=False)

        positive_similarities = []
        negative_similarities = []

        for i in range(len(questions)):
            pos_sim = cosine_similarity(
                question_embeddings[i].reshape(1, -1),
                positive_embeddings[i].reshape(1, -1),
            )[0, 0]
            positive_similarities.append(pos_sim)

            neg_sim = cosine_similarity(
                question_embeddings[i].reshape(1, -1),
                negative_embeddings[i].reshape(1, -1),
            )[0, 0]
            negative_similarities.append(neg_sim)

        pos_sims = np.array(positive_similarities)
        neg_sims = np.array(negative_similarities)

        avg_positive_similarity = float(np.mean(pos_sims))
        avg_negative_similarity = float(np.mean(neg_sims))
        similarity_ratio = (
            avg_positive_similarity / avg_negative_similarity
            if avg_negative_similarity > 0
            else float("inf")
        )

        expected_scores = [1] * len(positive_similarities) + [0] * len(
            negative_similarities
        )
        actual_scores = positive_similarities + negative_similarities

        if len(set(actual_scores)) > 1:
            correlation_result = spearmanr(expected_scores, actual_scores)
            correlation_value = correlation_result[0]  # type: ignore
            spearman_corr = (
                float(correlation_value) if not np.isnan(correlation_value) else 0.0  # type: ignore
            )
        else:
            spearman_corr = 0.0

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

        return {"trained": trained_metrics, "baseline": baseline_metrics}
