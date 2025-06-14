"""Dataset validation utilities for evaluation."""

from pathlib import Path

import pandas as pd

from vectorize.training.exceptions import DatasetValidationError

__all__ = ["DatasetValidator"]


class DatasetValidator:
    """Handles dataset validation for evaluation."""

    REQUIRED_COLUMNS = {"Question", "Positive", "Negative"}

    @classmethod
    def validate_dataset(cls, dataset_path: Path) -> pd.DataFrame:
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

        missing_cols = cls.REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            raise DatasetValidationError(
                f"Missing columns in {dataset_path}: {missing_cols}"
            )

        if df.empty:
            raise DatasetValidationError(f"Dataset {dataset_path} is empty")

        for col in cls.REQUIRED_COLUMNS:
            if df[col].isnull().any() is True:
                raise DatasetValidationError(
                    f"Column '{col}' contains null values in {dataset_path}"
                )

        return df
