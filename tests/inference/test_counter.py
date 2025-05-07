# ruff: noqa: S101

"""Tests for inference endpoint."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

_MODEL_NAME = "pytorch_model"
_BAD_MODEL_NAME = "nonexistent_model"


@pytest.mark.asyncio
@pytest.mark.inference
class TestEmbeddings:
    """Tests for the embeddings endpoint."""

    @classmethod
    async def test_basic_embedding(cls, client: TestClient) -> None:
        """Test basic embedding generation with a simple input."""
        payload = {"model": _MODEL_NAME, "input": "This is a test sentence."}

        # Check inference counter
        counter_response = client.get(f"/embeddings/counter/{_MODEL_NAME}")
        assert counter_response.status_code == status.HTTP_200_OK
        first_key = next(iter(counter_response.json()))
        current_count = counter_response.json()[first_key]

        response = client.post("/embeddings", json=payload)
        assert response.status_code == status.HTTP_200_OK

        # Check inference counter
        counter_response = client.get(f"/embeddings/counter/{_MODEL_NAME}")
        assert counter_response.status_code == status.HTTP_200_OK
        first_key = next(iter(counter_response.json()))
        assert counter_response.json()[first_key] == current_count + 1

        response = client.post("/embeddings", json=payload)
        assert response.status_code == status.HTTP_200_OK

        # Check inference counter
        counter_response = client.get(f"/embeddings/counter/{_MODEL_NAME}")
        assert counter_response.status_code == status.HTTP_200_OK
        first_key = next(iter(counter_response.json()))
        assert counter_response.json()[first_key] == current_count + 2

    @classmethod
    async def test_faulty_embedding(cls, client: TestClient) -> None:
        """Test faulty embedding generation with a non-existent model."""
        payload = {"model": _BAD_MODEL_NAME, "input": "This is a test sentence."}

        # Check inference counter
        counter_response = client.get(f"/embeddings/counter/{_BAD_MODEL_NAME}")
        assert counter_response.status_code == status.HTTP_404_NOT_FOUND
        first_key = next(iter(counter_response.json()))
        current_count = counter_response.json()[first_key]

        response = client.post("/embeddings", json=payload)
        assert response.status_code == status.HTTP_404_NOT_FOUND

        # Check inference counter
        counter_response = client.get(f"/embeddings/counter/{_BAD_MODEL_NAME}")
        assert counter_response.status_code == status.HTTP_404_NOT_FOUND
        first_key = next(iter(counter_response.json()))
        assert counter_response.json()[first_key] == current_count

        response = client.post("/embeddings", json=payload)
        assert response.status_code == status.HTTP_404_NOT_FOUND

        # Check inference counter
        counter_response = client.get(f"/embeddings/counter/{_BAD_MODEL_NAME}")
        assert counter_response.status_code == status.HTTP_404_NOT_FOUND
        first_key = next(iter(counter_response.json()))
        assert counter_response.json()[first_key] == current_count
