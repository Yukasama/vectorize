"""Similarity computation utilities for evaluation."""


import numpy as np
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

__all__ = ["SimilarityCalculator"]


class SimilarityCalculator:
    """Handles similarity computations for evaluation."""

    @staticmethod
    def compute_cosine_similarities(
        model: SentenceTransformer,
        questions: list[str],
        positives: list[str],
        negatives: list[str],
    ) -> tuple[list[float], list[float]]:
        """Compute cosine similarities for question-positive-negative triplets.

        Args:
            model: SentenceTransformer model
            questions: List of question texts
            positives: List of positive answer texts
            negatives: List of negative answer texts

        Returns:
            Tuple of (positive_similarities, negative_similarities)
        """
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

        return positive_similarities, negative_similarities

    @staticmethod
    def compute_spearman_correlation(
        positive_similarities: list[float],
        negative_similarities: list[float],
    ) -> float:
        """Compute Spearman correlation for similarity ranking.

        Args:
            positive_similarities: List of positive similarities
            negative_similarities: List of negative similarities

        Returns:
            Spearman correlation coefficient
        """
        expected_scores = [1] * len(positive_similarities) + [
            0
        ] * len(negative_similarities)
        actual_scores = positive_similarities + negative_similarities

        if len(set(actual_scores)) > 1:
            correlation_result = spearmanr(expected_scores, actual_scores)
            correlation_value = correlation_result[0]  # type: ignore[index]
            return (
                float(correlation_value) if not np.isnan(correlation_value) else 0.0  # type: ignore[arg-type]
            )
        return 0.0

    @staticmethod
    def compute_similarity_ratio(
        avg_positive: float,
        avg_negative: float,
    ) -> float:
        """Compute similarity ratio (positive/negative).

        Args:
            avg_positive: Average positive similarity
            avg_negative: Average negative similarity

        Returns:
            Similarity ratio or infinity if negative is zero
        """
        return (
            avg_positive / avg_negative
            if avg_negative > 0
            else float("inf")
        )
