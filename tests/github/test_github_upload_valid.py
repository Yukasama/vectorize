"""Test für das Laden eines gültigen GitHub-Modells."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

_GITHUB_URL = "https://github.com/facebookresearch/nougat"
_TAG = "main"


@pytest.mark.github
def test_load_nougat_model(client: TestClient) -> None:
    """Testet das Laden des Facebook nougat Modells von GitHub."""
    response = client.post(
        "/uploads/github",
        json={"github_url": _GITHUB_URL, "tag": _TAG},
    )
    assert response.status_code == status.HTTP_201_CREATED
