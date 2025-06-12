"""Tests for evaluation functionality."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from vectorize.evaluation.evaluation import EvaluationMetrics, TrainingEvaluator
from vectorize.training.exceptions import DatasetValidationError


class TestEvaluationMetrics:
    """Tests for EvaluationMetrics class."""

    def test_metrics_initialization(self):
        """Test that metrics are initialized correctly."""
        metrics = EvaluationMetrics(
            avg_positive_similarity=0.8,
            avg_negative_similarity=0.3,
            similarity_ratio=2.67,
            spearman_correlation=0.7,
            num_samples=100
        )

        assert metrics.avg_positive_similarity == 0.8
        assert metrics.avg_negative_similarity == 0.3
        assert metrics.similarity_ratio == 2.67
        assert metrics.spearman_correlation == 0.7
        assert metrics.num_samples == 100

    def test_is_training_successful_positive(self):
        """Test successful training detection."""
        metrics = EvaluationMetrics(
            avg_positive_similarity=0.8,
            avg_negative_similarity=0.3,
            similarity_ratio=2.67,  # > 1.2 and pos > neg
            spearman_correlation=0.7,
            num_samples=100
        )
        assert metrics.is_training_successful() is True

    def test_is_training_successful_negative(self):
        """Test unsuccessful training detection."""
        metrics = EvaluationMetrics(
            avg_positive_similarity=0.3,
            avg_negative_similarity=0.8,
            similarity_ratio=0.375,  # < 1.2 and pos < neg
            spearman_correlation=0.2,
            num_samples=100
        )
        assert metrics.is_training_successful() is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = EvaluationMetrics(
            avg_positive_similarity=0.8,
            avg_negative_similarity=0.3,
            similarity_ratio=2.67,
            spearman_correlation=0.7,
            num_samples=100
        )

        result = metrics.to_dict()
        expected_keys = {
            "avg_positive_similarity",
            "avg_negative_similarity",
            "similarity_ratio",
            "spearman_correlation",
            "num_samples",
            "is_training_successful"
        }
        assert set(result.keys()) == expected_keys
        assert result["is_training_successful"] is True


class TestTrainingEvaluator:
    """Tests for TrainingEvaluator class."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        data = {
            "Question": [
                "What is machine learning?",
                "How does AI work?",
                "What is deep learning?"
            ],
            "Positive": [
                "Machine learning is a subset of artificial intelligence",
                "AI works by processing data through algorithms",
                "Deep learning uses neural networks with multiple layers"
            ],
            "Negative": [
                "The weather is sunny today",
                "Pizza is delicious food",
                "Cars need fuel to run"
            ]
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def temp_dataset_file(self, sample_dataset):
        """Create a temporary JSONL file for testing."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonl', delete=False, encoding='utf-8'
        ) as f:
            sample_dataset.to_json(f.name, orient='records', lines=True)
            return Path(f.name)

    def test_validate_dataset_valid(self, temp_dataset_file):
        """Test dataset validation with valid data."""
        # Use a dummy model path for initialization
        evaluator = TrainingEvaluator("dummy/path")
        df = evaluator._validate_dataset(temp_dataset_file)

        assert len(df) == 3
        assert set(df.columns) >= {"Question", "Positive", "Negative"}

    def test_validate_dataset_missing_columns(self):
        """Test dataset validation with missing columns."""
        # Create dataset with missing column
        data = {"Question": ["test"], "Positive": ["test"]}  # Missing Negative
        df = pd.DataFrame(data)

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonl', delete=False, encoding='utf-8'
        ) as f:
            df.to_json(f.name, orient='records', lines=True)
            temp_file = Path(f.name)

        evaluator = TrainingEvaluator("dummy/path")

        with pytest.raises(DatasetValidationError, match="Missing columns"):
            evaluator._validate_dataset(temp_file)

    def test_validate_dataset_empty_file(self):
        """Test dataset validation with completely empty file."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonl', delete=False, encoding='utf-8'
        ) as f:
            # Write nothing - completely empty file
            temp_file = Path(f.name)

        evaluator = TrainingEvaluator("dummy/path")

        # Empty file should trigger missing columns error
        with pytest.raises(DatasetValidationError, match="Missing columns"):
            evaluator._validate_dataset(temp_file)

    def test_validate_dataset_null_values(self):
        """Test dataset validation with null values."""
        data = {
            "Question": ["test", None],
            "Positive": ["test", "test"],
            "Negative": ["test", "test"]
        }
        df = pd.DataFrame(data)

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonl', delete=False, encoding='utf-8'
        ) as f:
            df.to_json(f.name, orient='records', lines=True)
            temp_file = Path(f.name)

        evaluator = TrainingEvaluator("dummy/path")

        with pytest.raises(DatasetValidationError, match="contains null values"):
            evaluator._validate_dataset(temp_file)


@pytest.mark.integration
class TestEvaluationIntegration:
    """Integration tests that require actual models (marked for optional execution)."""

    @pytest.mark.skip(reason="Requires actual model - run manually if needed")
    def test_full_evaluation_pipeline(self, temp_dataset_file):
        """Test full evaluation pipeline with real model."""
        # This test would require a real model path
        # model_path = "sentence-transformers/all-MiniLM-L6-v2"
        # evaluator = TrainingEvaluator(model_path)

        # This would perform actual evaluation
        # metrics = evaluator.evaluate_dataset(temp_dataset_file, max_samples=2)
        # assert isinstance(metrics, EvaluationMetrics)
        # assert metrics.num_samples == 2
