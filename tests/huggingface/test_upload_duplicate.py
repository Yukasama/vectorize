# ruff: noqa: S101

"""Test für das Laden eines bereits vorhandenen Huggingface-Modells."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

_MODEL_ID = "distilbert-base-uncased"
_TAG = "main"


@pytest.mark.huggingface
def test_load_distilbert_model(client: TestClient) -> None:
    """Testet das Laden eines bereits vorhandenen distilbert-base-uncased Modells."""
    response = client.post(
        "/uploads/huggingface", json={"model_id": _MODEL_ID, "tag": _TAG}
    )
    assert response.status_code == status.HTTP_409_CONFLICT
