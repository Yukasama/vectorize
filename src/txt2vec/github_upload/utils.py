"""
Utilities for validating GitHub repository URLs.
"""

import re


class GitHubUtils:
    """
    Utility methods for working with GitHub repository URLs.

    This class provides validation for GitHub repo URLs and can be
    extended with additional GitHub-related helper methods (e.g., for
    fetching releases, interacting with the GitHub API, etc.).
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
        """
        Check if the given URL is a valid GitHub repository URL.

        Args:
            url: The URL string to validate.

        Returns:
            True if `url` matches the GitHub repo pattern, False otherwise.
        """
        return re.match(GitHubUtils.GITHUB_URL_REGEX, url) is not None

    @staticmethod
    def parse_github_url(url: str) -> tuple[str, str]:
        """
        Extract the owner and repository name from a valid GitHub URL.

        Args:
            url: The GitHub URL string.

        Returns:
            A tuple (owner, repository) if the URL is valid.

        Raises:
            ValueError: If the URL is not a valid GitHub repository URL.
        """
        match = re.match(GitHubUtils.GITHUB_URL_REGEX, url)
        if match:
            return match.group("owner"), match.group("repo")
        raise ValueError("Invalid GitHub repository URL.")
