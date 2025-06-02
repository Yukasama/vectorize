"""Upload module for vectorize."""

from vectorize.upload.exceptions import (
    EmptyModelError,
    InvalidModelError,
    InvalidUrlError,
    InvalidZipError,
    ModelAlreadyExistsError,
    ModelTooLargeError,
    NoValidModelsFoundError,
)
from vectorize.upload.github_service import (
    load_github_model_and_cache_only,
    repo_info,
)
from vectorize.upload.huggingface_service import (
    load_model_and_cache_only,
)
from vectorize.upload.local_service import upload_zip_model
from vectorize.upload.models import UploadTask
from vectorize.upload.repository import (
    save_upload_task,
    update_upload_task_status,
)
from vectorize.upload.router import router
from vectorize.upload.schemas import (
    GitHubModelRequest,
    HuggingFaceModelRequest,
)
from vectorize.upload.tasks import (
    process_github_model_background,
    process_huggingface_model_background,
)

__all__ = [
    "EmptyModelError",
    "GitHubModelRequest",
    "HuggingFaceModelRequest",
    "InvalidModelError",
    "InvalidUrlError",
    "InvalidZipError",
    "ModelAlreadyExistsError",
    "ModelTooLargeError",
    "NoValidModelsFoundError",
    "UploadTask",
    "load_github_model_and_cache_only",
    "load_model_and_cache_only",
    "process_github_model_background",
    "process_huggingface_model_background",
    "repo_info",
    "router",
    "save_upload_task",
    "update_upload_task_status",
    "upload_zip_model",
]
