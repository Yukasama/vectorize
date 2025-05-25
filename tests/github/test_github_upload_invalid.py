"""Github endpoint invalid input checks."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

# Cases where Pydantic HttpUrl (or your str→url check) will reject outright
_INVALID_SYNTAX_URLS = [
    "",                    # leer
    "not a url",           # kein valider URL-String
    "ftp://github.com/owner/repo",  # nosec  # NOSONAR #ungültiges Schema
]

# Cases where it *is* a valid HTTP URL, but not a well-formed GitHub repo
_INVALID_GITHUB_URLS = [
    "http://example.com/owner/repo",  # nosec  # NOSONAR #ungültige url
    "https://github.com/",
]


@pytest.mark.github
@pytest.mark.parametrize("bad_url", _INVALID_SYNTAX_URLS)
def test_load_with_malformed_url_should_422(client: TestClient, bad_url: str) -> None:
    """Ungültige URL-Syntax führt zu 422 Unprocessable Entity."""
    response = client.post(
        "/uploads/github",
        json={"github_url": bad_url},
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY  # noqa: S101


@pytest.mark.github
@pytest.mark.parametrize("bad_url", _INVALID_GITHUB_URLS)
def test_load_with_non_github_url_should_400(client: TestClient, bad_url: str) -> None:
    """Korrekte HTTP-URL, aber kein GitHub-Repo → 400 Bad Request."""
    response = client.post(
        "/uploads/github",
        json={"github_url": bad_url},
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST  # noqa: S101
