# ruff: noqa: S101
"""Github endpoint invalid input checks."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient


@pytest.mark.github
@pytest.mark.parametrize("payload", [
    {},  # completely empty
    {"owner": "someuser"},  # missing repo_name
    {"repo_name": "somerepo"},  # missing owner
    {"owner": "", "repo_name": "somerepo"},  # invalid owner
    {"owner": "someuser", "repo_name": ""},  # invalid repo
])
def test_invalid_input_returns_422(client: TestClient, payload: dict) -> None:
    """Invalid payloads should result in 422 Unprocessable Entity."""
    response = client.post("/uploads/github", json=payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.github
@pytest.mark.parametrize("payload", [
    {"owner": "example", "repo_name": "nonexistent-repo"},
    {"owner": "example", "repo_name": "valid-repo", "tag": "nonexistent-branch"},
])
def test_nonexistent_github_repo_or_branch_returns_400(client: TestClient,
 payload: dict) -> None:
    """Valid input format but nonexistent repo/branch should return 400 Bad Request."""
    response = client.post("/uploads/github", json=payload)
    assert response.status_code == status.HTTP_404_NOT_FOUND
