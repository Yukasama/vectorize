# ruff: noqa: S101

"""Tests for AI Model GET endpoints."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

_WRONG_TAG = "non-existent-tag"
_VALID_TAG = "pytorch_model"


@pytest.mark.asyncio
@pytest.mark.ai_model
@pytest.mark.ai_model_read
class TestGetAIModels:
    """Tests for GET /models and GET /models/{ai_model_id} endpoints."""

    @classmethod
    async def test_get_model_by_id(cls, client: TestClient) -> None:
        """Test retrieving a single AI model by ID."""
        response = client.get(f"/models/{_VALID_TAG}")
        assert response.status_code == status.HTTP_200_OK

        model = response.json()
        assert "id" in model
        assert "name" in model
        assert model["model_tag"] == str(_VALID_TAG)
        assert "source" in model
        assert "created_at" in model
        assert "updated_at" in model
        assert "version" in model

        assert "ETag" in response.headers
        etag = response.headers["ETag"].strip('"')
        assert etag == str(model["version"])

    @classmethod
    async def test_get_model_with_matching_etag(cls, client: TestClient) -> None:
        """Test retrieving an AI model with a matching ETag."""
        response = client.get(f"/models/{_VALID_TAG}", headers={"If-None-Match": '"0"'})

        assert response.status_code == status.HTTP_304_NOT_MODIFIED
        assert response.content == b""

    @classmethod
    async def test_get_model_with_non_matching_etag(cls, client: TestClient) -> None:
        """Test retrieving an AI model with a non-matching ETag."""
        response = client.get(
            f"/models/{_VALID_TAG}", headers={"If-None-Match": '"wrong"'}
        )

        assert response.status_code == status.HTTP_200_OK
        model = response.json()
        assert model["model_tag"] == str(_VALID_TAG)

    @classmethod
    async def test_get_model_non_existent_id(cls, client: TestClient) -> None:
        """Test retrieving an AI model with a non-existent ID."""
        response = client.get(f"/models/{_WRONG_TAG}")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert response.json()["code"] == "NOT_FOUND"
