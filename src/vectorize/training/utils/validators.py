"""Centralized training data validation for SBERT training pipeline."""

from pathlib import Path

import pandas as pd

from ..exceptions import DatasetValidationError


class TrainingDataValidator:
    REQUIRED_COLUMNS = {"Question", "Positive", "Negative"}

    @classmethod
    def validate_dataset(cls, dataset_path: Path) -> pd.DataFrame:
        """Zentrale Validierung für alle Datasets.

        Args:
            dataset_path (Path): Pfad zur JSONL-Datei.

        Returns:
            pd.DataFrame: Validiertes DataFrame.

        Raises:
            DatasetValidationError: Bei ungültigen oder inkonsistenten Daten.
        """
        try:
            df = pd.read_json(dataset_path, lines=True)
        except Exception as exc:
            raise DatasetValidationError(f"Invalid JSONL file {dataset_path}: {exc}")

        missing_cols = cls.REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            raise DatasetValidationError(f"Missing columns in {dataset_path}: {missing_cols}")

        if df.empty:
            raise DatasetValidationError(f"Dataset {dataset_path} is empty")

        if df.isnull().any().any():
            raise DatasetValidationError(f"Dataset {dataset_path} contains null values")

        return df
