"""
Service for importing models
"""

from loguru import logger

import httpx
from fastapi import HTTPException
from upload.utils import GitHubUtils


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
                "Model file not found on GitHub: repo=%s/%s, file=%s, status=%d, message=%s",
                owner,
                repo,
                file_path,
                resp.status_code,
                resp.text,
            )
            raise HTTPException(status_code=404, detail="Model file not found.")
        else:
            logger.error(
                "Unexpected GitHub API error: repo=%s/%s, file=%s, status=%d, message=%s",
                owner,
                repo,
                file_path,
                resp.status_code,
                resp.text,
            )
            raise HTTPException(status_code=500, detail="GitHub API error.")

        save_path = ""  # Not saving for now — placeholder

        # TODO: Decide where and how to save models in dev/prod environments
        # model_dir = f"./models/{owner}_{repo}"
        # os.makedirs(model_dir, exist_ok=True)
        # save_path = os.path.join(model_dir, file_path)
        # file_resp = await client.get(download_url)
        # with open(save_path, "wb") as f:
        #     f.write(file_resp.content)

    return {
        "message": f"Model would be saved to {save_path} (saving disabled for now)."
    }
