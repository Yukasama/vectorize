# ruff: noqa: S101

"""Test für das Laden eines ungültigen Huggingface-Modells."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

_INVALID_URL = "https://github.com/facebookresearch/nougat"  # TODO change to invalid and adjust json
_TAG = "main"


@pytest.mark.github
def test_load_invalid_model_should_fail(client: TestClient) -> None:
    """Testet, dass das Laden eines ungültigen Modells fehlschlägt."""
    response = client.post("/uploads/github", json={"model_id": _MODEL_ID, "tag": _TAG})
    assert response.status_code == status.HTTP_404_NOT_FOUND
