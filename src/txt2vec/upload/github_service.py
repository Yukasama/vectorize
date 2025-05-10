"""Service for importing models."""

from http import HTTPStatus
from pathlib import Path

import httpx
from loguru import logger

from txt2vec.upload.exceptions import (
    GitHubApiError,
    InvalidGitHubUrlError,
    MissingDownloadUrlError,
    ModelDownloadError,
    NoValidModelsFoundError,
)
from txt2vec.upload.utils import GitHubUtils


async def handle_model_download(github_url: str) -> dict:
    """Handles downloading a PyTorch model file from a GitHub repository.

    Args:
        github_url (str): A valid GitHub repository URL.

    Returns:
        dict: A message indicating the result of the operation.

    Raises:
        HTTPException: If the URL is invalid, the file is not found,
                       or a GitHub API error occurs.

    """
    if not GitHubUtils.is_github_url(github_url):
        raise InvalidGitHubUrlError()

    owner, repo = GitHubUtils.parse_github_url(github_url)
    file_path = "pytorch_model.bin"

    async with httpx.AsyncClient() as client:
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
        meta_resp = await client.get(api_url)

        if meta_resp.status_code == HTTPStatus.OK:
            download_url = meta_resp.json().get("download_url")
            if not download_url:
                raise MissingDownloadUrlError()

            file_resp = await client.get(download_url)
            if file_resp.status_code != HTTPStatus.OK:
                logger.error(
                    "Failed to download model: repo={}/{} url={} status={} message={}",
                    owner,
                    repo,
                    download_url,
                    file_resp.status_code,
                    file_resp.text,
                )
            raise ModelDownloadError()
            save_dir = Path("models") / f"{owner}_{repo}"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / file_path
            save_path.write_bytes(file_resp.content)

        if meta_resp.status_code == HTTPStatus.NOT_FOUND:
            logger.error(
                "Model file not found on GitHub: repo={}/{} file={}",
                owner,
                repo,
                file_path,
            )
            raise NoValidModelsFoundError()
        logger.error(
            "Unexpected GitHub API error: repo={}/{} file={} status={} message={}",
            owner,
            repo,
            file_path,
            meta_resp.status_code,
            meta_resp.text,
        )
        raise GitHubApiError()

    return {"message": f"Model downloaded and saved to `{save_path}`"}
