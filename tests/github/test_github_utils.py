"""Unit tests for GitHubUtils URL validation."""

import pytest

from txt2vec.upload.utils import GitHubUtils


@pytest.mark.github
@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/owner/repo",
        "http://github.com/owner/repo",
        "https://www.github.com/owner/repo",
        "https://github.com/owner-name/repo_name",
        "https://github.com/owner/repo.git",
        "https://github.com/owner/repo/",
    ],
)
def test_is_github_url_valid(url: str) -> None:
    """Ensure that valid GitHub repository URLs are recognized as such.

    Valid formats covered:
      - HTTP or HTTPS schemes
      - Optional 'www' subdomain
      - Owner/org names with hyphens or dots
      - Repo names with hyphens or underscores
      - Optional '.git' suffix
      - Optional trailing slash
    """
    assert GitHubUtils.is_github_url(url), f"Expected valid: {url}"  # noqa: S101


@pytest.mark.parametrize(
    "url",
    [
        "https://gitlab.com/owner/repo",
        "https://github.com/owner",  # missing repo
        "https://github.com/owner/repo/extra",  # too many segments
        "ftp://github.com/owner/repo",  # wrong scheme
        "github.com/owner/repo",  # missing scheme
        "https://github.com//repo",  # missing owner
        "https://github.com/owner/",  # missing repo
        "https://notgithub.com/owner/repo",  # wrong domain
    ],
)
def test_is_github_url_invalid(url: str) -> None:
    """Ensure that non GitHub or malformed URLs are correctly rejected.

    Invalid cases include:
      - Wrong domains (e.g., gitlab.com or notgithub.com)
      - Missing repository segment
      - Extra path segments beyond owner/repo
      - Unsupported schemes (e.g., FTP)
      - URLs without HTTP/HTTPS scheme
    """
    assert not GitHubUtils.is_github_url(url), f"Expected invalid: {url}"  # noqa: S101
