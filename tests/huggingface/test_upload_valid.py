# ruff: noqa: S101

"""Test für das Hochladen eines Huggingface-Modells."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

_MODEL_ID = "distilbert-base-uncased"
_MODEL_SECOND_ID = "distilroberta-base"
_TAG = "main"


@pytest.mark.huggingface
def test_load_distilbert_model(client: TestClient) -> None:
    """Testet das Laden des distilbert-base-uncased Modells von Huggingface."""
    response = client.post(
        "/uploads/huggingface", json={"model_id": _MODEL_ID, "tag": _TAG}
    )
    assert response.status_code == status.HTTP_201_CREATED


@pytest.mark.huggingface
def test_load_distilbert_model_without_tag(client: TestClient) -> None:
    """Testet das Laden des distilbert-base-uncased Modells von Huggingface ohne Tag."""
    response = client.post("/uploads/huggingface", json={"model_id": _MODEL_SECOND_ID})
    assert response.status_code == status.HTTP_201_CREATED
