# ruff: noqa: S101

"""Test für das Hochladen eines HuggingFace-Modells."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

_MODEL_TAG = "distilbert-base-uncased"
_MODEL_SECOND_TAG = "distilroberta-base"
_REVISION = "main"


@pytest.mark.huggingface
def test_load_distilbert_model(client: TestClient) -> None:
    """Testet das Laden des distilbert-base-uncased Modells von HuggingFace."""
    response = client.post(
        "/uploads/huggingface", json={"model_tag": _MODEL_TAG, "revision": _REVISION}
    )
    assert response.status_code == status.HTTP_201_CREATED


@pytest.mark.huggingface
def test_load_distilbert_model_without_revision(client: TestClient) -> None:
    """Testet das Laden des distilbert-base-uncased Modells von HuggingFace.

    ohne Revision.
    """
    response = client.post(
        "/uploads/huggingface",
        json={"model_tag": _MODEL_SECOND_TAG},
    )
    assert response.status_code == status.HTTP_201_CREATED
