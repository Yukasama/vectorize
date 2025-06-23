"""Dataset validation utilities for evaluation."""

from pathlib import Path

import pandas as pd
from fastapi import status

from vectorize.common.app_error import AppError
from vectorize.config.errors import ErrorCode


class EvaluationDatasetValidationError(AppError):
    """Exception raised when the evaluation dataset is invalid."""

    error_code = ErrorCode.INVALID_FILE
    status_code = status.HTTP_400_BAD_REQUEST

    def __init__(self, msg: str) -> None:
        """Initialize with a validation error message."""
        super().__init__(f"Evaluation dataset validation failed: {msg}")


__all__ = ["DatasetValidator", "EvaluationDatasetValidationError"]


class DatasetValidator:
    """Handles dataset validation for evaluation."""

    REQUIRED_COLUMNS = {"question", "positive", "negative"}

    @classmethod
    def validate_dataset(cls, dataset_path: Path) -> pd.DataFrame:
        """Validate and load dataset for evaluation.

        Args:
            dataset_path: Path to JSONL dataset file

        Returns:
            Validated DataFrame with required columns

        Raises:
            EvaluationDatasetValidationError: If dataset is invalid or cannot be loaded
        """
        if not dataset_path.exists():
            raise EvaluationDatasetValidationError(
                f"Dataset file does not exist: {dataset_path}"
            )

        if not dataset_path.is_file():
            raise EvaluationDatasetValidationError(
                f"Path is not a file: {dataset_path}"
            )

        try:
            df = pd.read_json(dataset_path, lines=True)
        except Exception as exc:
            raise EvaluationDatasetValidationError(
                f"Invalid JSONL file {dataset_path}: {exc}"
            ) from exc

        missing_cols = cls.REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            raise EvaluationDatasetValidationError(
                f"Missing columns in {dataset_path}: {missing_cols}"
            )

        if df.empty:
            raise EvaluationDatasetValidationError(f"Dataset {dataset_path} is empty")

        for col in cls.REQUIRED_COLUMNS:
            if bool(df[col].isnull().any()):
                null_count = df[col].isnull().sum()
                raise EvaluationDatasetValidationError(
                    f"Column '{col}' contains {null_count} null values "
                    f"in {dataset_path}"
                )

        for col in cls.REQUIRED_COLUMNS:
            empty_count = (~df[col].astype(str).str.strip().astype(bool)).sum()
            if empty_count > 0:
                raise EvaluationDatasetValidationError(
                    f"Column '{col}' contains {empty_count} empty values "
                    f"in {dataset_path}"
                )

        return df
