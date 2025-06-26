# ruff: noqa: S101

"""Tests for evaluation functionality."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from vectorize.evaluation.evaluation import EvaluationMetrics
from vectorize.evaluation.utils import DatasetValidator
from vectorize.evaluation.utils.dataset_validator import (
    EvaluationDatasetValidationError,
)


@pytest.mark.evaluation
class TestEvaluationMetrics:
    """Tests for EvaluationMetrics class."""

    AVG_POSITIVE_SIMILARITY = 0.8
    AVG_NEGATIVE_SIMILARITY = 0.3
    SIMILARITY_RATIO = 2.67
    SPEARMAN_CORRELATION = 0.7
    NUM_SAMPLES = 100

    @staticmethod
    def test_metrics_initialization() -> None:
        """Test that metrics are initialized correctly."""
        metrics = EvaluationMetrics(
            avg_positive_similarity=TestEvaluationMetrics.AVG_POSITIVE_SIMILARITY,
            avg_negative_similarity=TestEvaluationMetrics.AVG_NEGATIVE_SIMILARITY,
            similarity_ratio=TestEvaluationMetrics.SIMILARITY_RATIO,
            spearman_correlation=TestEvaluationMetrics.SPEARMAN_CORRELATION,
            num_samples=TestEvaluationMetrics.NUM_SAMPLES,
        )

        assert metrics.avg_positive_similarity == (
            TestEvaluationMetrics.AVG_POSITIVE_SIMILARITY
        )
        assert metrics.avg_negative_similarity == (
            TestEvaluationMetrics.AVG_NEGATIVE_SIMILARITY
        )
        assert metrics.similarity_ratio == TestEvaluationMetrics.SIMILARITY_RATIO
        assert metrics.spearman_correlation == (
            TestEvaluationMetrics.SPEARMAN_CORRELATION
        )
        assert metrics.num_samples == TestEvaluationMetrics.NUM_SAMPLES

    @staticmethod
    def test_to_dict() -> None:
        """Test conversion to dictionary."""
        metrics = EvaluationMetrics(
            avg_positive_similarity=TestEvaluationMetrics.AVG_POSITIVE_SIMILARITY,
            avg_negative_similarity=TestEvaluationMetrics.AVG_NEGATIVE_SIMILARITY,
            similarity_ratio=TestEvaluationMetrics.SIMILARITY_RATIO,
            spearman_correlation=TestEvaluationMetrics.SPEARMAN_CORRELATION,
            num_samples=TestEvaluationMetrics.NUM_SAMPLES,
        )

        result = metrics.to_dict()
        expected_keys = {
            "avg_positive_similarity",
            "avg_negative_similarity",
            "similarity_ratio",
            "spearman_correlation",
            "num_samples",
        }
        assert set(result.keys()) == expected_keys


@pytest.mark.evaluation
class TestTrainingEvaluator:
    """Tests for TrainingEvaluator class."""

    @staticmethod
    @pytest.fixture
    def sample_dataset() -> pd.DataFrame:
        """Create a sample dataset for testing."""
        data = {
            "question": [
                "What is machine learning?",
                "How does AI work?",
                "What is deep learning?",
            ],
            "positive": [
                "Machine learning is a subset of artificial intelligence",
                "AI works by processing data through algorithms",
                "Deep learning uses neural networks with multiple layers",
            ],
            "negative": [
                "The weather is sunny today",
                "Pizza is delicious food",
                "Cars need fuel to run",
            ],
        }
        return pd.DataFrame(data)

    @staticmethod
    @pytest.fixture
    def temp_dataset_file(sample_dataset: pd.DataFrame) -> Path:
        """Create a temporary JSONL file for testing."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            sample_dataset.to_json(f.name, orient="records", lines=True)
            return Path(f.name)

    @staticmethod
    def test_validate_dataset_valid(temp_dataset_file: Path) -> None:
        """Test dataset validation with valid data."""
        df = DatasetValidator.validate_dataset(temp_dataset_file)

        expected_len = 3
        assert len(df) == expected_len
        assert set(df.columns) >= {"question", "positive", "negative"}

    @staticmethod
    def test_validate_dataset_missing_columns() -> None:
        """Test dataset validation with missing columns."""
        data = {"Question": ["test"], "Positive": ["test"]}  # Missing Negative
        df = pd.DataFrame(data)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            df.to_json(f.name, orient="records", lines=True)
            temp_file = Path(f.name)

        with pytest.raises(EvaluationDatasetValidationError, match="Missing columns"):
            DatasetValidator.validate_dataset(temp_file)

    @staticmethod
    def test_validate_dataset_empty_file() -> None:
        """Test dataset validation with completely empty file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            temp_file = Path(f.name)

        with pytest.raises(EvaluationDatasetValidationError, match="Missing columns"):
            DatasetValidator.validate_dataset(temp_file)

    @staticmethod
    def test_validate_dataset_null_values() -> None:
        """Test dataset validation with null values."""
        data = {
            "question": ["test", None],
            "positive": ["test", "test"],
            "negative": ["test", "test"],
        }
        df = pd.DataFrame(data)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            df.to_json(f.name, orient="records", lines=True)
            temp_file = Path(f.name)

        with pytest.raises(
            EvaluationDatasetValidationError, match=r"contains \d+ null values"
        ):
            DatasetValidator.validate_dataset(temp_file)
