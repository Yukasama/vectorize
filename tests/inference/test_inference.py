# ruff: noqa: S101

"""Tests for inference endpoint."""

from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from txt2vec.errors import ErrorCode

_MODEL_NAME = "pytorch_model"
_BAD_MODEL_NAME = "nonexistent_model"

VALID_INPUTS = [
    "This is a simple test sentence.",
    "Multiple languages are supported: こんにちは, 你好, مرحبا, 안녕하세요!",
    "AAAAAAAAAA " * 100,
    "",
]

_HUGE_DIMENSIONS = 1000000


@pytest.mark.asyncio
@pytest.mark.inference
class TestEmbeddings:
    """Tests for the embeddings endpoint."""

    @staticmethod
    async def _get_embeddings(
        client: TestClient,
        payload: dict[str, Any],
        expected_status: int = status.HTTP_200_OK,
        expected_error: ErrorCode = None,
    ) -> dict[str, Any]:
        """Make a request to the embeddings endpoint and verify response."""
        response = client.post("/embeddings", json=payload)
        assert response.status_code == expected_status

        if expected_error:
            assert response.json()["code"] == expected_error
            return {}

        return response.json()

    async def test_basic_embedding(self, client: TestClient) -> None:
        """Test basic embedding generation with a simple input."""
        payload = {"model": _MODEL_NAME, "input": "This is a test sentence."}

        response = await self._get_embeddings(client, payload)

        # Verify core response structure
        assert response["object"] == "list"
        assert response["model"] == _MODEL_NAME
        assert "data" in response
        assert len(response["data"]) == 1

        # Check embedding structure
        embedding_data = response["data"][0]
        assert "embedding" in embedding_data
        assert "index" in embedding_data
        assert isinstance(embedding_data["embedding"], list)
        assert len(embedding_data["embedding"]) > 0

        # Check usage statistics
        assert "usage" in response
        assert "prompt_tokens" in response["usage"]
        assert "total_tokens" in response["usage"]
        assert response["usage"]["prompt_tokens"] > 0

    @pytest.mark.parametrize("input_text", VALID_INPUTS)
    async def test_various_inputs(self, client: TestClient, input_text: str) -> None:
        """Test embedding generation with various input texts."""
        payload = {"model": _MODEL_NAME, "input": input_text}

        response = await self._get_embeddings(client, payload)
        assert response["object"] == "list"
        assert len(response["data"]) == 1

    async def test_multiple_inputs(self, client: TestClient) -> None:
        """Test batch embedding generation with multiple inputs."""
        payload = {"model": _MODEL_NAME, "input": VALID_INPUTS}

        response = await self._get_embeddings(client, payload)

        assert response["object"] == "list"
        assert len(response["data"]) == len(VALID_INPUTS)

        for i, item in enumerate(response["data"]):
            assert item["index"] == i

    async def test_with_dimensions_parameter(self, client: TestClient) -> None:
        """Test embedding generation with custom output dimensions."""
        dimensions = 10
        payload = {
            "model": _MODEL_NAME,
            "input": "This is a test sentence.",
            "dimensions": dimensions,
        }

        response = await self._get_embeddings(client, payload)

        embedding = response["data"][0]["embedding"]
        assert len(embedding) == dimensions

    async def test_token_id_input(self, client: TestClient) -> None:
        """Test embedding generation with token IDs instead of text."""
        payload = {
            "model": _MODEL_NAME,
            "input": [[0, 5, 8, 12, 23, 45, 2]],
        }

        response = await self._get_embeddings(client, payload)
        assert "data" in response
        assert len(response["data"]) > 0

    async def test_nonexistent_model(self, client: TestClient) -> None:
        """Test embedding generation with non-existent model."""
        payload = {"model": _BAD_MODEL_NAME, "input": "This is a test sentence."}

        await self._get_embeddings(
            client,
            payload,
            expected_status=status.HTTP_404_NOT_FOUND,
            expected_error=ErrorCode.NOT_FOUND,
        )

    async def test_invalid_input_type(self, client: TestClient) -> None:
        """Test embedding generation with invalid input type."""
        payload = {
            "model": _MODEL_NAME,
            "input": 12345,
        }

        await self._get_embeddings(
            client,
            payload,
            expected_status=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    async def test_missing_required_field(self, client: TestClient) -> None:
        """Test embedding generation with missing required fields."""
        payload = {"model": _MODEL_NAME}

        await self._get_embeddings(
            client,
            payload,
            expected_status=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

        payload = {"input": "This is a test sentence."}

        await self._get_embeddings(
            client,
            payload,
            expected_status=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    async def test_invalid_dimensions(self, client: TestClient) -> None:
        """Test embedding generation with invalid dimensions parameter."""
        payload = {
            "model": _MODEL_NAME,
            "input": "This is a test sentence.",
            "dimensions": -10,
        }

        await self._get_embeddings(
            client,
            payload,
            expected_status=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    async def test_extremely_large_dimensions(self, client: TestClient) -> None:
        """Test embedding generation with extremely large dimensions parameter."""
        payload = {
            "model": _MODEL_NAME,
            "input": "This is a test sentence.",
            "dimensions": 100000,
        }

        response = await self._get_embeddings(client, payload)

        if "data" in response:
            assert len(response["data"]) > 0
            # The dimension should be capped at model's max dimension
            assert len(response["data"][0]["embedding"]) < _HUGE_DIMENSIONS
