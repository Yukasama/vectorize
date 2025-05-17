# ruff: noqa: S101

"""Test für das Laden eines ungültigen Huggingface-Modells."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

_MODEL_ID = "nonexistent-model-id-xyz1234567890"
_TAG = "main"


@pytest.mark.huggingface
def test_load_invalid_model_should_fail(client: TestClient) -> None:
    """Testet, dass das Laden eines ungültigen Modells fehlschlägt."""
    response = client.post(
        "/uploads/huggingface", json={"model_id": _MODEL_ID, "tag": _TAG}
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND
