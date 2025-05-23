"""Utilities for validating GitHub repository URLs."""

import re
from urllib.parse import urlparse

from pydantic import HttpUrl

from txt2vec.upload.exceptions import InvalidUrlError


class GitHubUtils:
    # only anchor the base part; allow anything after
    GITHUB_BASE_REGEX = (
        r"^https?://(?:www\.)?github\.com/"
        r"(?P<owner>[\w\-\.]+)/"
        r"(?P<repo>[\w\-\.]+)"
    )

    @staticmethod
    def is_github_url(url: str | HttpUrl) -> bool:
        return bool(re.match(GitHubUtils.GITHUB_BASE_REGEX, str(url)))

    @staticmethod
    def parse_github_url(
        url: str | HttpUrl
    ) -> tuple[str, str, str | None]:
        """Returns (owner, repo, maybe_revision).

        If the path contains /releases/tag/{tag}, that tag is returned as revision.
        Otherwise revision is None.
        """
        text = str(url)
        m = re.match(GitHubUtils.GITHUB_BASE_REGEX, text)
        if not m:
            raise InvalidUrlError()

        owner = m.group("owner")
        repo = m.group("repo")

        path_parts = urlparse(text).path.strip("/").split("/")
        revision = None
        if len(path_parts) >= 4 and path_parts[2] == "releases" and path_parts[3] == "tag":
            revision = path_parts[4]

        return owner, repo, revision
