"""Service for importing models."""


import shutil
import tempfile

import git
import httpx

from txt2vec.ai_model.exceptions import ModelNotFoundError
from txt2vec.common.status import TaskStatus
from txt2vec.upload import repository
from txt2vec.upload.exceptions import (
    ServiceUnavailableError,
)


def repo_info(repo_url: str, revision: str = None):
    # Check if repo/branch/tag exists on GitHub (HEAD or API request)
    api_url = str(repo_url).replace("https://github.com/", "https://api.github.com/repos/").rstrip("/")
    branch = revision or "main"
    check_url = f"{api_url}/branches/{branch}"
    resp = httpx.get(check_url, timeout=10)
    if resp.status_code != 200:  # TODO use status.200
        raise ModelNotFoundError(repo_url, branch)
    return True


def load_model_and_cache_only(repo_url: str, revision: str = None) -> str:
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
    try:
        model_dir = load_model_and_cache_only(repo_url, tag)
        # ggf. Modell speichern, falls notwendig (eigene Methode implementieren)
        await repository.update_upload_task_status(db, task_id, TaskStatus.COMPLETED)
    except Exception as e:
        await repository.update_upload_task_status(db, task_id, TaskStatus.FAILED, error_msg=str(e))
