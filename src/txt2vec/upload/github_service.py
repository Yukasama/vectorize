"""Service for importing models from GitHub into the application cache.

This module provides functions to check repository info, clone and cache models, and
process the import in a background task.
"""

import shutil
import tempfile

import git
import httpx
from fastapi import status

from txt2vec.ai_model.exceptions import ModelNotFoundError
from txt2vec.common.status import TaskStatus
from txt2vec.upload import repository
from txt2vec.upload.exceptions import (
    ServiceUnavailableError,
)


def repo_info(repo_url: str, revision: str = None):
    """Check if a GitHub repository and branch/tag exists.

    This function verifies the existence of the specified repository and branch (or tag)
    by querying the GitHub API. Raises ModelNotFoundError
    if the branch/tag is not found.

    Args:
          repo_url (str): The HTTPS URL of the GitHub repository
          (e.g., "https://github.com/user/repo").
        revision (str, optional): The branch or tag name to check. Defaults to 'main'.

    Returns:
        bool: True if the repository and branch/tag exist.

    Raises:
        ModelNotFoundError: If the repository or specified branch/tag
        is not found on GitHub.
    """
    # Check if repo/branch/tag exists on GitHub (HEAD or API request)
    api_url = (
        str(repo_url)
        .replace("https://github.com/", "https://api.github.com/repos/")
        .rstrip("/")
    )
    branch = revision or "main"
    check_url = f"{api_url}/branches/{branch}"
    resp = httpx.get(check_url, timeout=10)
    if resp.status_code != status.HTTP_200_OK:
        raise ModelNotFoundError(repo_url, branch)
    return True


def load_model_and_cache_only(repo_url: str, revision: str = None) -> str:
    """Clone a GitHub repository and cache the model locally.

    This function creates a temporary directory, clones the specified repository
    at the given branch or tag (with depth=1), and returns the local directory path.
    On failure, cleans up and raises ServiceUnavailableError.

    Args:
        repo_url (str): The HTTPS URL of the GitHub repository.
        revision (str, optional): The branch or tag to checkout. Defaults to 'main'.

    Returns:
        str: The path to the temporary directory containing the cloned repository.

    Raises:
        ServiceUnavailableError: If cloning or checkout fails for any reason.
    """
    # - Clone repo (with GitPython)
    # - Checkout correct revision/tag/branch
    # - Validate model structure (check for config.json, model file etc.)
    # - Return local path
    tmp_dir = tempfile.mkdtemp(prefix="github_model_")
    try:
        git.Repo.clone_from(repo_url, tmp_dir, branch=revision or "main", depth=1)
        return tmp_dir
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise ServiceUnavailableError from e


async def process_github_model_background(repo_url, tag, task_id, db):
    """Background task to process and update the status of a GitHub model import.

    This coroutine clones and caches the model, then updates the upload task status
    in the database to COMPLETED or FAILED based on the outcome.

    Args:
        repo_url (str): The HTTPS URL of the GitHub repository.
        tag (str): The branch or tag to checkout.
        task_id (int): The identifier of the upload task in the database.
        db: The database session or connection object.

    Returns:
        None
    """
    try:
        model_dir = load_model_and_cache_only(repo_url, tag)
        # ggf. Modell speichern, falls notwendig (eigene Methode implementieren)
        await repository.update_upload_task_status(db, task_id, TaskStatus.COMPLETED)
    except Exception as e:
        await repository.update_upload_task_status(
            db, task_id, TaskStatus.FAILED, error_msg=str(e)
        )
