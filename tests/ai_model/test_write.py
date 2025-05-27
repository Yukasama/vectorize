# ruff: noqa: S101

"""Tests for AI model PUT (update) endpoint."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

_VALID_ID = "7d2f3e4b-8c7f-4d2a-9f1e-0a6f3e4d2a5b"
_VALID_TAG = "pytorch_model"
_INVALID_ID = "8d2f3e4b-8c7f-4d2a-9f1e-0a6f3e4d2a5b"
_NON_EXISTENT_ID = "12345678-1234-5678-1234-567812345678"
_DELETE_ID = "2d2f3e4b-8c7f-4d2a-9f1e-0a6f3e4d2a5b"
_DELETE_TAG = "huge_model"


@pytest.mark.asyncio
@pytest.mark.ai_model
@pytest.mark.ai_model_write
class TestUpdateAIModels:
    """Tests for the PUT /models/{ai_model_id} endpoint."""

    @classmethod
    async def test_successful_update(cls, client: TestClient) -> None:
        """Test successfully updating an AI model with valid data and matching ETag."""
        update_data = {"name": "Updated AI Model Name"}
        response = client.put(
            f"/models/{_VALID_ID}", json=update_data, headers={"If-Match": '"0"'}
        )

        assert response.status_code == status.HTTP_204_NO_CONTENT
        assert "Location" in response.headers
        assert response.headers["Location"].endswith(str(_VALID_ID))

        assert "ETag" in response.headers
        assert response.headers["ETag"] == '"1"'

        get_response = client.get(f"/models/{_VALID_TAG}")
        assert get_response.status_code == status.HTTP_200_OK
        updated_model = get_response.json()
        assert updated_model["name"] == "Updated AI Model Name"

    @classmethod
    async def test_failed_update_missing_name(cls, client: TestClient) -> None:
        """Test failed update when required field 'name' is missing."""
        update_data = {}
        response = client.put(
            f"/models/{_INVALID_ID}", json=update_data, headers={"If-Match": '"0"'}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        response_body = response.json()
        assert response_body["detail"][0]["loc"] == ["body", "name"]
        assert response_body["detail"][0]["msg"] == "Field required"

    @classmethod
    async def test_version_mismatch(cls, client: TestClient) -> None:
        """Test failed update when ETag doesn't match current version (lost update)."""
        update_data = {"name": "This Update Should Fail"}
        response = client.put(
            f"/models/{_INVALID_ID}",
            json=update_data,
            headers={"If-Match": '"999"'},  # Incorrect ETag
        )

        assert response.status_code == status.HTTP_412_PRECONDITION_FAILED
        response_body = response.json()
        assert response_body["code"] == "VERSION_MISMATCH"

        assert "ETag" in response.headers
        assert response.headers["ETag"] == '"0"'

    @classmethod
    async def test_missing_if_match_header(cls, client: TestClient) -> None:
        """Test failed update when If-Match header is not provided."""
        update_data = {"name": "This Update Should Fail Too"}
        response = client.put(
            f"/models/{_INVALID_ID}",
            json=update_data,
            # No If-Match header
        )

        assert response.status_code == status.HTTP_428_PRECONDITION_REQUIRED
        response_body = response.json()
        assert response_body["code"] == "VERSION_MISSING"

    @classmethod
    async def test_delete(cls, client: TestClient) -> None:
        """Test successful deletion of a model."""
        response = client.get("/models?size=100")
        assert response.status_code == status.HTTP_200_OK
        models_length = len(response.json()["items"])

        response = client.delete(f"/models/{_DELETE_ID}")
        assert response.status_code == status.HTTP_204_NO_CONTENT

        model = client.get(f"/models/{_DELETE_TAG}")
        assert model.status_code == status.HTTP_404_NOT_FOUND

        response = client.get("/models?size=100")
        assert len(response.json()["items"]) == models_length - 1

    @classmethod
    async def test_delete_not_exist(cls, client: TestClient) -> None:
        """Test deletion of a non-existent model."""
        response = client.get("/models?size=100")
        assert response.status_code == status.HTTP_200_OK
        models_length = len(response.json()["items"])

        response = client.delete(f"/models/{_NON_EXISTENT_ID}")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        response_body = response.json()
        assert response_body["code"] == "NOT_FOUND"

        response = client.get("/models?size=100")
        assert len(response.json()["items"]) == models_length
