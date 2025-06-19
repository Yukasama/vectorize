"""Centralized training data validation for SBERT training pipeline."""

from pathlib import Path

import pandas as pd

from vectorize.evaluation.utils.dataset_validator import DatasetValidator

__all__ = ["TrainingDataValidator"]


class TrainingDataValidator:
    """Validates training data for SBERT training pipeline.

    This is a thin wrapper around the centralized DatasetValidator
    to maintain backwards compatibility in the training module.
    """

    REQUIRED_COLUMNS = {"question", "positive", "negative"}

    @classmethod
    def validate_dataset(cls, dataset_path: Path) -> pd.DataFrame:
        """Central validation for all datasets.

        Args:
            dataset_path (Path): Path to the JSONL file.

        Returns:
            pd.DataFrame: Validated DataFrame.

        Raises:
            DatasetValidationError: For invalid or inconsistent data.
        """
        # Delegate to the centralized DatasetValidator
        return DatasetValidator.validate_dataset(dataset_path)
