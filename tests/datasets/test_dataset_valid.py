# ruff: noqa: S101

"""Tests for valid datasets."""

import asyncio
import json
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlmodel.ext.asyncio.session import AsyncSession

from txt2vec.datasets.repository import get_dataset

from .utils import get_test_file

_TRAINING_FOLDER = "test_data"
_VALID_FOLDER = "valid"
_MALICIOUS_FOLDER = "malicious"

_DEFAULT = "default"
_CUSTOM_FORMAT = "custom_fields.csv"
_INFER_FORMAT = "infer_fields.csv"

_NULL_BYTE_INJECTION = "%00nullbyte.csv"
_COMMAND_INJECTION = "; rm -rf %2F.csv"


@pytest.mark.asyncio
@pytest.mark.dataset
@pytest.mark.dataset_valid
class TestValidDatasets:
    """Tests for valid dataset uploads."""

    _base_dir = Path(__file__).parent.parent.parent / _TRAINING_FOLDER / "datasets"
    valid_dir = _base_dir / _VALID_FOLDER
    malicious_dir = _base_dir / _MALICIOUS_FOLDER

    @staticmethod
    async def _upload_and_verify(
        client: TestClient,
        session: AsyncSession,
        file_path: Path,
        extra_data: dict[str, Any] | None = None,
    ) -> UUID:
        """Upload a file and verify the dataset is created."""
        files = get_test_file(file_path)

        response = client.post("/datasets", files=files, data=extra_data or {})
        assert response.status_code == status.HTTP_201_CREATED

        dataset_id = response.headers["Location"].split("/")[-1]
        assert dataset_id is not None

        max_retries = 3
        for attempt in range(max_retries):
            try:
                dataset = await get_dataset(db=session, dataset_id=UUID(dataset_id))
                assert dataset.id == UUID(dataset_id)
                return dataset.id
            except Exception:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.5)

        # This shouldn't be reached, but satisfies type checker
        raise AssertionError("Failed to verify dataset")

    @pytest.mark.parametrize("ext", ["csv", "json", "xml", "xlsx"])
    async def test_dataset_formats_upload(
        self, ext: str, session: AsyncSession, client: TestClient
    ) -> None:
        """Parametrized test for uploading multiple file formats."""
        test_file_path = self.valid_dir / f"{_DEFAULT}.{ext}"
        await self._upload_and_verify(client, session, test_file_path)

    async def test_dataset_custom_fields(
        self, client: TestClient, session: AsyncSession
    ) -> None:
        """Test uploading a dataset with custom fields."""
        test_file_path = self.valid_dir / _CUSTOM_FORMAT

        column_mapping = {
            "question": "q",
            "positive": "answer",
            "negative": "no_context",
        }

        await self._upload_and_verify(
            client, session, test_file_path, {"options": json.dumps(column_mapping)}
        )

    async def test_dataset_infer_fields(
        self, client: TestClient, session: AsyncSession
    ) -> None:
        """Test uploading a dataset with inferred fields."""
        test_file_path = self.valid_dir / _INFER_FORMAT
        await self._upload_and_verify(client, session, test_file_path)

    @pytest.mark.parametrize("file_name", [_NULL_BYTE_INJECTION, _COMMAND_INJECTION])
    async def test_malicious_files(
        self,
        client: TestClient,
        session: AsyncSession,
        file_name: str,
    ) -> None:
        """Test uploading a file with no name."""
        test_file_path = self.malicious_dir / file_name
        await self._upload_and_verify(client, session, test_file_path)
