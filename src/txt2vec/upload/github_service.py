"""Service for importing github models."""

import tempfile
from pathlib import Path

import httpx
from fastapi import status
from git import Repo
from loguru import logger

from txt2vec.ai_model.exceptions import ModelNotFoundError
from txt2vec.upload.exceptions import (
    InvalidModelError,
    InvalidUrlError,
    NoValidModelsFoundError,
)
from txt2vec.upload.utils import GitHubUtils

_github_models: dict[str, object] = {}


def load_github_model_and_cache_only(github_url: str) -> None:
    """Clone ein GitHub-Repo, suche das Model und cache es lokal + im Memory.

    Args:
        github_url: HTTPS-URL zum GitHub-Repo mit Model (z.B. owner/repo).

    Returns:
        dict: Metadaten zum gecachten Model (owner, repo, branch, local_path, key).

    Raises:
        InvalidUrlError: Wenn die URL kein GitHub-Repo ist.
        NoValidModelsFoundError: Wenn kein gültiges Model-File gefunden wurde.
        InvalidModelError: Für alle anderen Fehler während Clone/Load.
    """
    if not GitHubUtils.is_github_url(github_url):
        raise InvalidUrlError()
    owner, repo, url_tag = GitHubUtils.parse_github_url(github_url)
    branch = url_tag or "main"
    key = f"{owner}/{repo}@{branch}"

    if key in _github_models:
        logger.info("GitHub-Model bereits im Cache.", modelKey=key)
        return {"owner": owner, "repo": repo, "branch": branch,
                 "local_path": _github_models[key], "key": key}

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            Repo.clone_from(
                f"https://github.com/{owner}/{repo}.git",
                tmpdir,
            )
            repo_obj = Repo(tmpdir)
            repo_obj.git.checkout(branch)

            base = Path(tmpdir)
            candidates = list(base.rglob("*.safetensors"))
            if len(candidates) != 1:
                raise NoValidModelsFoundError(
                    f"{len(candidates)} Model-Dateien gefunden")

            model_file = candidates[0]
            logger.debug("Gefundene Model-Datei: %s", model_file)

            cache_dir = Path("github_model_cache") / f"{owner}_{repo}@{branch}"
            cache_dir.parent.mkdir(parents=True, exist_ok=True)
            if cache_dir.exists():
                logger.info("Model bereits lokal gecached unter %s", cache_dir)
            else:
                logger.info("Cache-Verzeichnis anlegen: %s", cache_dir)
                cache_dir.mkdir()
                dest = cache_dir / model_file.name
                model_file.replace(dest)
                model_file = dest

        logger.info("GitHub-Model erfolgreich gecached.", modelKey=key)

        return {
            "owner": owner,
            "repo": repo,
            "branch": branch,
            "local_path": str(model_file),
            "key": key,
        }

    except NoValidModelsFoundError:
        raise
    except Exception as e:
        logger.exception("Fehler beim Laden des GitHub-Models.", modelKey=key)
        raise InvalidModelError(f"GitHub-Import fehlgeschlagen: {e}") from e


def repo_info(repo_url: str, revision: str | None = None) -> bool:
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
