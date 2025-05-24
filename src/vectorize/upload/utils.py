"""Utilities for validating GitHub repository URLs."""

import re
from urllib.parse import urlparse

from pydantic import HttpUrl

from .exceptions import InvalidUrlError

_MIN_PATH_PARTS_FOR_RELEASE_URL = 4


__all__ = ["GitHubUtils"]


class GitHubUtils:
    """Utils supporting the github service.

    Raises:
        InvalidUrlError: _description_

    Returns:
        _type_: _description_
    """

    GITHUB_BASE_REGEX = (
        r"^https?://(?:www\.)?github\.com/"
        r"(?P<owner>[\w\-\.]+)/"
        r"(?P<repo>[\w\-\.]+)"
        r"(?:\.git)?/?$"
    )

    @staticmethod
    def is_github_url(url: str | HttpUrl) -> bool:
        """Checks if the provided Url is a github Url.

        Args:
            url (str | HttpUrl): _description_

        Returns:
            bool: _description_
        """
        return bool(re.fullmatch(GitHubUtils.GITHUB_BASE_REGEX, str(url)))

    @staticmethod
    def parse_github_url(url: str | HttpUrl) -> tuple[str, str, str | None]:
        """Returns (owner, repo, maybe_revision).

        If the path contains /releases/tag/{tag}, that tag is returned as revision.
        Otherwise revision is None.
        """
        text = str(url)
        m = re.fullmatch(GitHubUtils.GITHUB_BASE_REGEX, text)
        if not m:
            raise InvalidUrlError()

        owner = m.group("owner")
        repo = m.group("repo")

        path_parts = urlparse(text).path.strip("/").split("/")
        revision = None
        if (
            len(path_parts) >= _MIN_PATH_PARTS_FOR_RELEASE_URL
            and path_parts[2] == "releases"
            and path_parts[3] == "tag"
        ):
            revision = path_parts[4]

        return owner, repo, revision
