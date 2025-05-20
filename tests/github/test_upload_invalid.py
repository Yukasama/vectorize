"""Test für das Laden eines ungültigen GitHub-Modells."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

_INVALID_URL = "https://github.com/facebookresearch/this-model-does-not-exist"
_TAG = "main"


@pytest.mark.github
def test_load_invalid_model_should_fail(client: TestClient) -> None:
    """Testet, dass das Laden eines ungültigen Modells fehlschlägt."""
    response = client.post(
        "/uploads/github",
        json={"github_url": _INVALID_URL, "tag": _TAG},
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "not found" in response.json().get("detail", "").lower()
