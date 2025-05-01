"""Utilities for validating GitHub repository URLs.
"""

import re

from loguru import logger


class GitHubUtils:
    """Utility methods for working with GitHub repository URLs.
    """

    GITHUB_URL_REGEX = (
        r"^(https?://)"  # http:// or https://
        r"(www\.)?"  # optional www.
        r"github\.com/"  # github.com/
        r"(?P<owner>[\w\-\.]+)/"  # owner or organization
        r"(?P<repo>[\w\-\.]+)"  # repository name
        r"(?:\.git)?"  # optional .git suffix
        r"/?$"  # optional trailing slash
    )

    @staticmethod
    def is_github_url(url: str) -> bool:
        """Check if the given URL is a valid GitHub repository URL.
        """
        logger.trace("Validating GitHub URL: {}", url)
        match = re.match(GitHubUtils.GITHUB_URL_REGEX, url)

        if match:
            logger.debug("Valid GitHub URL detected: {}", url)
            return True
        logger.debug("Invalid GitHub URL: {}", url)
        return False

    @staticmethod
    def parse_github_url(url: str) -> tuple[str, str]:
        """Extract the owner and repository name from a valid GitHub URL.
        """
        logger.trace("Parsing GitHub URL: {}", url)
        match = re.match(GitHubUtils.GITHUB_URL_REGEX, url)
        if match:
            owner, repo = match.group("owner"), match.group("repo")
            logger.debug("Parsed GitHub URL → owner: {}, repo: {}", owner, repo)
            return owner, repo

        logger.error("Failed to parse GitHub URL: {}", url)
        raise ValueError("Invalid GitHub repository URL.")
