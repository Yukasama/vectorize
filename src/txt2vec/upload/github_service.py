"""
Service for importing models
"""

from loguru import logger

import httpx
from fastapi import HTTPException
from txt2vec.upload.utils import GitHubUtils


async def handle_model_download(github_url: str) -> dict:
    """
    Handles downloading a PyTorch model file from a GitHub repository.

    Args:
        github_url (str): A valid GitHub repository URL.

    Returns:
        dict: A message indicating the result of the operation.

    Raises:
        HTTPException: If the URL is invalid, the file is not found,
                       or a GitHub API error occurs.
    """
    if not GitHubUtils.is_github_url(github_url):
        raise HTTPException(status_code=400, detail="Invalid GitHub URL.")

    owner, repo = GitHubUtils.parse_github_url(github_url)
    file_path = "pytorch_model.bin"

    async with httpx.AsyncClient() as client:
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
        resp = await client.get(api_url)

        if resp.status_code == 200:
            download_url = resp.json().get("download_url")
        elif resp.status_code == 404:
            logger.error(
                "Model file not found on GitHub: repo={}/{} file={} status={} message={}",
                owner,
                repo,
                file_path,
                resp.status_code,
                resp.text,
            )
            # FIXME: use logger.bind(...) for structured logging instead
            raise HTTPException(status_code=404, detail="Model file not found.")
        else:
            logger.error(
                "Unexpected GitHub API error: repo={}/{} file={} status={} message={}",
                owner,
                repo,
                file_path,
                resp.status_code,
                resp.text,
            )
            # FIXME: use logger.bind(...) for structured logging instead
            raise HTTPException(status_code=500, detail="GitHub API error.")

        save_path = ""  # Not saving for now — placeholder

    return {
        "message": f"Model would be saved to {save_path} (saving disabled for now)."
    }
