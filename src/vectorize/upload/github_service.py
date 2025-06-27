"""Service for importing GitHub models."""

import tempfile
from pathlib import Path

import httpx
from fastapi import status
from git import Repo
from loguru import logger

from .exceptions import InvalidModelError, ModelNotFoundError, NoValidModelsFoundError

__all__ = ["load_github_model_and_cache_only_svc", "repo_info"]

_github_models: dict[str, str] = {}


async def load_github_model_and_cache_only_svc(  # noqa: RUF029 NOSONAR
    owner: str,
    repo: str,
    branch: str = "main",
) -> dict[str, str]:
    """Loads a GitHub repo and caches the model.

    Clones a GitHub repository, looks for exactly one each of
    `pytorch_model.bin`, `config.json`, and `tokenizer.json`, then caches
    them both on disk and in memory.

    Args:
        owner: GitHub username or organization (e.g., "domoar").
        repo:  Repository name (e.g., "BogusModel").
        branch: Branch or tag name (default: "main").

    Returns:
        dict: {
            "owner": owner,
            "repo": repo,
            "branch": branch,
            "local_path": "<path to pytorch_model.bin>",
            "key": "owner/repo@branch"
        }

    Raises:
        NoValidModelsFoundError: If the repo does not contain exactly one
            of each required file.
        InvalidModelError: For any other error (clone failure, I/O issues, etc.).
    """
    key = f"{owner}/{repo}@{branch}"

    if key in _github_models:
        logger.info("GitHub-Model already cached", modelKey=key)
        return {
            "owner": owner,
            "repo": repo,
            "branch": branch,
            "local_path": _github_models[key],
            "key": key,
        }

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_url = f"https://github.com/{owner}/{repo}.git"
            Repo.clone_from(repo_url, tmpdir, branch=branch)
            repo_obj = Repo(tmpdir)
            repo_obj.git.checkout(branch)

            base = Path(tmpdir)

            checks: list[tuple[str, str]] = [
                ("pytorch_model.bin", "Model-Datei"),
                ("config.json", "Config-Datei"),
                ("tokenizer.json", "Tokenizer-Datei"),
            ]

            paths: dict[str, Path] = {}
            for filename, description in checks:
                matches = list(base.rglob(filename))
                if len(matches) != 1:
                    raise NoValidModelsFoundError(
                        f"{len(matches)} {description} found for {filename}"
                    )
                paths[filename] = matches[0]
                logger.debug("Found {}: {}", description, matches[0])

            cache_dir = Path("gh_cache") / f"{owner}_{repo}@{branch}"
            cache_dir_parent = cache_dir.parent
            cache_dir_parent.mkdir(parents=True, exist_ok=True)

            if cache_dir.exists():
                logger.info("Model already cached {}", cache_dir)
            else:
                logger.info("Creating cache dir{}", cache_dir)
                cache_dir.mkdir()

                for filename, pfad in paths.items():
                    dest = cache_dir / filename
                    pfad.replace(dest)
                    paths[filename] = dest

            lokal_pfad = str((cache_dir / "pytorch_model.bin").resolve())
            _github_models[key] = lokal_pfad

        logger.info("Model cached", modelKey=key)
        return {
            "owner": owner,
            "repo": repo,
            "branch": branch,
            "local_path": lokal_pfad,
            "key": key,
        }

    except NoValidModelsFoundError:
        raise
    except Exception as e:
        logger.exception("Error loading model", modelKey=key)
        raise InvalidModelError(f"Model upload failed: {e}") from e


def repo_info(repo_url: str, revision: str | None = None) -> bool:
    """Check whether a GitHub repository and a specific branch / tag exist.

    Uses GitHub API endpoint: /repos/{owner}/{repo}/branches/{branch}

    Args:
        repo_url (str): HTTPS URL of the GitHub repo (e.g., "https://github.com/user/repo").
        revision (str | None): Branch or tag name. If None, defaults to "main".

    Returns:
        bool: True if both the repository and the given branch / tag exist.

    Raises:
        ModelNotFoundError: If the repository or the branch / tag cannot be found.
    """
    api_url = str(repo_url).replace(
        "https://github.com/", "https://api.github.com/repos/"
    ).rstrip("/")
    branch = revision or "main"
    check_url = f"{api_url}/branches/{branch}"
    resp = httpx.get(check_url, timeout=10)
    if resp.status_code != status.HTTP_200_OK:
        raise ModelNotFoundError(check_url)
    return True
