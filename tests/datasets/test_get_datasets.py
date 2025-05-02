# ruff: noqa: S101

"""Tests for dataset GET endpoints."""

from pathlib import Path
from uuid import UUID

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from .utils import get_test_file

_TRAINING_FOLDER = "test_data"
_VALID_FOLDER = "valid"
_DEFAULT_FILE = "default.csv"


@pytest.mark.asyncio
@pytest.mark.dataset
class TestGetDatasets:
    """Tests for GET /datasets and GET /datasets/{dataset_id} endpoints."""

    _base_dir = Path(__file__).parent.parent.parent / _TRAINING_FOLDER / "datasets"
    valid_dir = _base_dir / _VALID_FOLDER

    @staticmethod
    async def _create_test_dataset(client: TestClient) -> UUID:
        """Create a test dataset and return its ID."""
        file_path = (
            Path(__file__).parent.parent.parent
            / _TRAINING_FOLDER
            / "datasets"
            / _VALID_FOLDER
            / _DEFAULT_FILE
        )
        files = get_test_file(file_path)

        response = client.post("/datasets", files=files)
        assert response.status_code == status.HTTP_201_CREATED

        dataset_id = response.headers["Location"].split("/")[-1]
        return UUID(dataset_id)

    async def test_get_all_datasets(self, client: TestClient) -> None:
        """Test retrieving all datasets."""
        await self._create_test_dataset(client)

        response = client.get("/datasets")
        assert response.status_code == status.HTTP_200_OK

        datasets = response.json()
        assert isinstance(datasets, list)
        assert len(datasets) > 0

        for dataset in datasets:
            assert "id" in dataset
            assert "name" in dataset
            assert "classification" in dataset
            assert "created_at" in dataset
            assert "rows" in dataset
            assert "file_name" not in dataset
            assert "updated_at" not in dataset
            assert "synthesis_id" not in dataset

    async def test_get_dataset_by_id(self, client: TestClient) -> None:
        """Test retrieving a single dataset by ID."""
        dataset_id = await self._create_test_dataset(client)

        response = client.get(f"/datasets/{dataset_id}")
        assert response.status_code == status.HTTP_200_OK

        dataset = response.json()
        assert dataset["id"] == str(dataset_id)
        assert "name" in dataset
        assert "classification" in dataset
        assert "created_at" in dataset
        assert "updated_at" in dataset

        assert "ETAG" in response.headers
        etag = response.headers["ETAG"].strip('"')
        assert etag == "0"

    async def test_get_dataset_with_matching_etag(self, client: TestClient) -> None:
        """Test retrieving a dataset with a matching ETag."""
        dataset_id = await self._create_test_dataset(client)

        response = client.get(
            f"/datasets/{dataset_id}",
            headers={"If-None-Match": '"0"'},
        )

        assert response.status_code == status.HTTP_304_NOT_MODIFIED
        assert response.content == b""

    async def test_get_dataset_with_non_matching_etag(self, client: TestClient) -> None:
        """Test retrieving a dataset with a non-matching ETag."""
        dataset_id = await self._create_test_dataset(client)

        response = client.get(
            f"/datasets/{dataset_id}",
            headers={"If-None-Match": '"wrong"'},
        )

        assert response.status_code == status.HTTP_200_OK
        dataset = response.json()
        assert dataset["id"] == str(dataset_id)

    @classmethod
    async def test_get_dataset_non_existent_id(cls, client: TestClient) -> None:
        """Test retrieving a dataset with a non-existent ID."""
        non_existent_id = "00000000-0000-0000-0000-000000000000"

        response = client.get(f"/datasets/{non_existent_id}")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert response.json()["code"] == "NOT_FOUND"
