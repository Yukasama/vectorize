# ruff: noqa: S101

"""Tests for dataset GET endpoints."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

_WRONG_ID = "00000000-0000-0000-0000-000000000000"
_VALID_ID = "5d2f3e4b-8c7f-4d2a-9f1e-0a6f3e4d2a5b"


@pytest.mark.asyncio
@pytest.mark.dataset
@pytest.mark.dataset_read
class TestGetDatasets:
    """Tests for GET /datasets and GET /datasets/{dataset_id} endpoints."""

    @staticmethod
    def _assert_dataset_shape(dataset: dict) -> None:
        """Ensure a single dataset dict exposes only the public fields."""
        required = {"id", "name", "classification", "created_at", "rows"}
        forbidden = {"file_name", "updated_at", "synthesis_id"}

        assert required.issubset(dataset.keys())
        assert forbidden.isdisjoint(dataset.keys())

    @classmethod
    async def test_get_datasets_default(cls, client: TestClient) -> None:
        """Default call (no params) returns a Page wrapper with items."""
        response = client.get("/datasets")
        assert response.status_code == status.HTTP_200_OK

        page = response.json()
        assert isinstance(page, dict)
        assert set(page.keys()) == {"items", "total", "limit", "offset"}

        items = page["items"]
        assert isinstance(items, list)
        assert len(items) > 0
        cls._assert_dataset_shape(items[0])

    @classmethod
    async def test_get_datasets_limit(cls, client: TestClient) -> None:
        """?limit constrains number of returned items and is echoed back."""
        limit = 2
        response = client.get(f"/datasets?limit={limit}")
        assert response.status_code == status.HTTP_200_OK

        page = response.json()
        assert page["limit"] == limit
        assert len(page["items"]) <= limit

    @classmethod
    async def test_get_datasets_offset_pagination(cls, client: TestClient) -> None:
        """Offset should skip the first *offset* rows; pages must not overlap."""
        limit = 3
        first_page = client.get(f"/datasets?limit={limit}&offset=0").json()
        second_page = client.get(f"/datasets?limit={limit}&offset={limit}").json()

        first_ids = {d["id"] for d in first_page["items"]}
        second_ids = {d["id"] for d in second_page["items"]}
        assert first_ids.isdisjoint(second_ids)

        assert len(first_ids | second_ids) <= first_page["total"]

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

        assert "ETag" in response.headers
        etag = response.headers["ETag"].strip('"')
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
