# ruff: noqa: S101

"""Tests for dataset GET endpoints."""

from pathlib import Path

import pytest
from fastapi import status
from fastapi.testclient import TestClient

_TRAINING_FOLDER = "test_data"
_VALID_FOLDER = "valid"

_WRONG_ID = "00000000-0000-0000-0000-000000000000"
_VALID_ID = "5d2f3e4b-8c7f-4d2a-9f1e-0a6f3e4d2a5b"


@pytest.mark.asyncio
@pytest.mark.dataset
class TestGetDatasets:
    """Tests for GET /datasets and GET /datasets/{dataset_id} endpoints."""

    _base_dir = Path(__file__).parent.parent.parent / _TRAINING_FOLDER / "datasets"
    valid_dir = _base_dir / _VALID_FOLDER

    @classmethod
    async def test_get_all_datasets(cls, client: TestClient) -> None:
        """Test retrieving all datasets."""
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

    @classmethod
    async def test_get_dataset_by_id(cls, client: TestClient) -> None:
        """Test retrieving a single dataset by ID."""
        response = client.get(f"/datasets/{_VALID_ID}")
        assert response.status_code == status.HTTP_200_OK

        dataset = response.json()
        assert dataset["id"] == str(_VALID_ID)
        assert "name" in dataset
        assert "classification" in dataset
        assert "created_at" in dataset
        assert "updated_at" in dataset

        assert "ETAG" in response.headers
        etag = response.headers["ETAG"].strip('"')
        assert etag == "0"

    @classmethod
    async def test_get_dataset_with_matching_etag(cls, client: TestClient) -> None:
        """Test retrieving a dataset with a matching ETag."""
        response = client.get(
            f"/datasets/{_VALID_ID}", headers={"If-None-Match": '"0"'}
        )

        assert response.status_code == status.HTTP_304_NOT_MODIFIED
        assert response.content == b""

    @classmethod
    async def test_get_dataset_with_non_matching_etag(cls, client: TestClient) -> None:
        """Test retrieving a dataset with a non-matching ETag."""
        response = client.get(
            f"/datasets/{_VALID_ID}", headers={"If-None-Match": '"wrong"'}
        )

        assert response.status_code == status.HTTP_200_OK
        dataset = response.json()
        assert dataset["id"] == str(_VALID_ID)

    @classmethod
    async def test_get_dataset_non_existent_id(cls, client: TestClient) -> None:
        """Test retrieving a dataset with a non-existent ID."""
        response = client.get(f"/datasets/{_WRONG_ID}")
        assert response.status_code == status.HTTP_404_NOT_FOUND
