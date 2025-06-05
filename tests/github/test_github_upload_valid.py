# ruff: noqa: S101
"""Github endpoint invalid input checks."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

_REPO_OWNER = "domoar"
_REPO_NAME = "BogusModel"
_BRANCH_DEFAULT = "main"
_BRANCH_DIFF = "DifferentBranch-Full"


@pytest.mark.github
def test_load_bogus_model_tag_main(client: TestClient) -> None:
    """Tests default main branch.

    Args:
        client (TestClient): _description_
    """
    payload = {
        "owner": _REPO_OWNER,
        "repo_name": _REPO_NAME,
        "tag": _BRANCH_DEFAULT
    }
    response = client.post("uploads/github", json=payload)
    assert response.status_code == status.HTTP_201_CREATED


@pytest.mark.github
def test_load_bogus_model_no_tag_conflict(client: TestClient) -> None:
    """Tests default with no tag branch.

    Args:
        client (TestClient): _description_
    """
    payload = {
        "owner": _REPO_OWNER,
        "repo_name": _REPO_NAME,
        "tag": ""
    }
    response = client.post("uploads/github", json=payload)
    assert response.status_code == status.HTTP_409_CONFLICT


@pytest.mark.github
def test_load_bogus_model_branch_tag(client: TestClient) -> None:
    """Tests default main branch.

    Args:
        client (TestClient): _description_
    """
    payload = {
        "owner": _REPO_OWNER,
        "repo_name": _REPO_NAME,
        "tag": _BRANCH_DIFF
    }
    response = client.post("uploads/github", json=payload)
    assert response.status_code == status.HTTP_201_CREATED
