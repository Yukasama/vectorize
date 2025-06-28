"""Training dataset validation utilities."""

import uuid
from pathlib import Path

import pandas as pd
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.config.config import settings
from vectorize.dataset.repository import get_dataset_db

from ..exceptions import InvalidDatasetIdError, TrainingDatasetNotFoundError


class TrainingDatasetValidator:
    """Validates datasets for training purposes."""

    @staticmethod
    async def validate_datasets(
        db: AsyncSession,
        train_dataset_ids: list[str],
        val_dataset_id: str | None = None,
    ) -> list[str]:
        """Validate and prepare dataset paths for training.

        Args:
            db: Database session
            train_dataset_ids: List of training dataset IDs
            val_dataset_id: Optional validation dataset ID

        Returns:
            List of validated dataset paths

        Raises:
            InvalidDatasetIdError: If dataset ID is invalid
            TrainingDatasetNotFoundError: If dataset file is not found or invalid
        """
        dataset_paths = []
        required_columns = {"question", "positive", "negative"}

        for train_ds_id in train_dataset_ids:
            path = await TrainingDatasetValidator._validate_single_dataset(
                db, train_ds_id, required_columns
            )
            dataset_paths.append(path)

        if val_dataset_id:
            path = await TrainingDatasetValidator._validate_single_dataset(
                db, val_dataset_id, required_columns
            )
            dataset_paths.append(path)

        TrainingDatasetValidator._check_files_exist(dataset_paths)
        return dataset_paths

    @staticmethod
    async def _validate_single_dataset(
        db: AsyncSession, dataset_id: str, required_columns: set[str]
    ) -> str:
        """Validate a single dataset by ID."""
        try:
            dataset_uuid = uuid.UUID(dataset_id)
        except ValueError as exc:
            raise InvalidDatasetIdError(dataset_id) from exc

        dataset = await get_dataset_db(db, dataset_uuid)
        if not dataset:
            raise TrainingDatasetNotFoundError(f"Dataset {dataset_id} not found")

        dataset_path = settings.dataset_upload_dir / dataset.file_name
        TrainingDatasetValidator._validate_jsonl_file(dataset_path, required_columns)
        return str(dataset_path)

    @staticmethod
    def _validate_jsonl_file(file_path: Path, required_columns: set[str]) -> None:
        """Validate a JSONL file and its columns.

        Args:
            file_path: Path to the JSONL file
            required_columns: Set of required column names

        Raises:
            TrainingDatasetNotFoundError: If file is invalid or missing columns
        """
        try:
            df = pd.read_json(file_path, lines=True)
        except Exception as exc:
            raise TrainingDatasetNotFoundError(
                f"Dataset {file_path} is not a valid JSONL file: {exc}"
            ) from exc

        if not required_columns.issubset(df.columns):
            missing_cols = required_columns - set(df.columns)
            raise TrainingDatasetNotFoundError(
                f"Dataset {file_path} is missing required columns: {missing_cols}"
            )

    @staticmethod
    def _check_files_exist(dataset_paths: list[str]) -> None:
        """Check that all dataset files exist."""
        missing = [str(p) for p in dataset_paths if not Path(p).is_file()]
        if missing:
            raise TrainingDatasetNotFoundError(
                f"Missing datasets: {', '.join(missing)}"
            )
