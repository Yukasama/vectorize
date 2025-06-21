# ruff: noqa: S101

"""Tests for synthesis endpoints."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

_INVALID_DATASET_ID = "abc12345-6789-0123-4567-89abcdef0123"
_MALFORMED_UUID = "not-a-valid-uuid"
_SYNTHESIS_MEDIA = "/synthesis/media"


@pytest.mark.asyncio
@pytest.mark.synthesis
class TestSynthesisTasks:
    """Tests for synthesis task endpoints."""

    @classmethod
    async def test_upload_media_with_invalid_dataset_id(
        cls, client: TestClient
    ) -> None:
        """Test creating synthesis task with non-existent dataset ID."""
        response = client.post(
            _SYNTHESIS_MEDIA, data={"dataset_id": _INVALID_DATASET_ID}
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND

    @classmethod
    async def test_upload_media_with_malformed_uuid(cls, client: TestClient) -> None:
        """Test creating synthesis task with malformed UUID."""
        response = client.post(_SYNTHESIS_MEDIA, data={"dataset_id": _MALFORMED_UUID})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @classmethod
    async def test_upload_media_without_files_or_dataset(
        cls, client: TestClient
    ) -> None:
        """Test creating synthesis task without providing files or dataset ID."""
        response = client.post(_SYNTHESIS_MEDIA)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert (
            "either files or existing dataset id must be provided"
            in response.json()["detail"].lower()
        )

    @classmethod
    async def test_get_synthesis_task_invalid_uuid(cls, client: TestClient) -> None:
        """Test retrieving synthesis task with invalid UUID."""
        invalid_uuid = "not-a-uuid"

        response = client.get(f"/synthesis/tasks/{invalid_uuid}")
        assert response.status_code in {
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_400_BAD_REQUEST,
        }

    @classmethod
    async def test_get_nonexistent_synthesis_task(cls, client: TestClient) -> None:
        """Test retrieving non-existent synthesis task."""
        nonexistent_id = "12345678-1234-5678-1234-567812345678"

        response = client.get(f"/synthesis/tasks/{nonexistent_id}")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "synthesis task not found" in response.json()["detail"].lower()

    @classmethod
    async def test_list_synthesis_tasks_endpoint_exists(
        cls, client: TestClient
    ) -> None:
        """Test that the list endpoint exists and returns proper format."""
        response = client.get("/synthesis")
        assert response.status_code == status.HTTP_200_OK

        tasks = response.json()
        assert isinstance(tasks, list)

    @classmethod
    async def test_list_synthesis_tasks_with_limit_parameter(
        cls, client: TestClient
    ) -> None:
        """Test listing synthesis tasks with limit parameter."""
        response = client.get("/synthesis?limit=5")
        assert response.status_code == status.HTTP_200_OK

        tasks = response.json()
        assert isinstance(tasks, list)

    @classmethod
    async def test_list_synthesis_tasks_with_invalid_limit(
        cls, client: TestClient
    ) -> None:
        """Test listing synthesis tasks with invalid limit parameter."""
        response = client.get("/synthesis?limit=-1")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @classmethod
    async def test_synthesis_router_registration(cls, client: TestClient) -> None:
        """Test that synthesis endpoints are properly registered."""
        response = client.post(_SYNTHESIS_MEDIA)
        assert response.status_code != status.HTTP_404_NOT_FOUND

        response = client.get("/synthesis")
        assert response.status_code == status.HTTP_200_OK

        response = client.get(f"/synthesis/tasks/{_INVALID_DATASET_ID}")
        assert response.status_code == status.HTTP_404_NOT_FOUND
