# ruff: noqa: S101

"""Test für das Laden eines ungültigen Huggingface-Modells."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

_MODEL_TAG = "nonexistent-model-id-xyz1234567890"
_REVISION = "main"


@pytest.mark.huggingface
def test_load_invalid_model_should_fail(client: TestClient) -> None:
    """Testet, dass das Laden eines ungültigen Modells fehlschlägt."""
    response = client.post(
        "/uploads/huggingface", json={"model_tag": _MODEL_TAG, "revision": _REVISION}
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND
