"""Router for model upload and management."""

from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse
from huggingface_hub import model_info
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError
from loguru import logger
from sqlmodel import Session, select
from sqlmodel.ext.asyncio.session import AsyncSession

from txt2vec.ai_model.exceptions import ModelNotFoundError
from txt2vec.ai_model.models import AIModel
from txt2vec.config.db import get_session
from txt2vec.datasets.exceptions import InvalidFileError
from txt2vec.upload import github_service, repository
from txt2vec.upload.exceptions import (
    ModelAlreadyExistsError,
    ServiceUnavailableError,
    UploadTaskNotFound,
)
from txt2vec.upload.models import (
    StatusResponse,
    UploadTask,
)
from txt2vec.upload.schemas import GitHubModelRequest, HuggingFaceModelRequest
from txt2vec.upload.zip_service import upload_zip_model

router = APIRouter(tags=["Model Upload"])


@router.post("/huggingface", status_code=status.HTTP_201_CREATED)
async def load_model_huggingface(
    data: HuggingFaceModelRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    """Upload a Hugging Face model by model_id and tag.

    Checks if the model already exists, verifies its presence on
    Hugging Face, creates an upload task, and starts background
    processing. Returns a 201 response with a Location header.

    Args:
        data: Model id and tag for Hugging Face.
        request: FastAPI request object.
        background_tasks: FastAPI background task manager.
        db: Async database session.

    Returns:
        Response with 201 status and Location header.
    """
    key = f"{data.model_id}@{data.tag}"

    model_exists = await db.exec(select(AIModel).where(AIModel.model_tag == key))
    if model_exists.first():
        raise ModelAlreadyExistsError(key)

    try:
        model_info(repo_id=data.model_id, revision=data.tag)
    except (EntryNotFoundError, HfHubHTTPError) as e:
        raise ModelNotFoundError(data.model_id, data.tag) from e
    except Exception as e:
        logger.exception("Fehler beim Laden:")
        raise ServiceUnavailableError from e


@router.post("/github", status_code=status.HTTP_201_CREATED)
async def upload_github_model(
    data: GitHubModelRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    # 1. Build model_tag
    key = f"{data.github_url}@{'main'}"  # 2. Check if model already exists
    model_exists = await db.exec(select(AIModel).where(AIModel.model_tag == key))
    if model_exists.first():
        raise ModelAlreadyExistsError(key)
    # 3. Check repo/tag existence
    try:
        github_service.repo_info(repo_url=data.github_url, revision="main")
    except ModelNotFoundError:
        raise
    except Exception as e:
        logger.exception(f"Fehler beim Check des GitHub-Repos: {e}")
        raise ServiceUnavailableError from e
    # 4. Create upload task, start background
    upload_task = await repository.create_upload_task(db, key, "GITHUB")
    background_tasks.add_task(
        github_service.process_github_model_background,
        data.github_url,
        "main",
        upload_task.id,
        db,
    )
    location_url = str(request.url_for("get_status", upload_id=upload_task.id))
    return Response(
        status_code=status.HTTP_201_CREATED, headers={"Location": location_url}
    )


@router.get("/upload/{upload_id}/status", response_model=StatusResponse)
def get_status(upload_id: str, session: Session = Depends(get_session)):
    """Poll the current state of an UploadTask."""
    task = session.get(UploadTask, upload_id)
    if task is None:
        raise UploadTaskNotFound()

    return StatusResponse(
        upload_id=task.id,
        status=task.task_status,
        error_msg=task.error_msg,
    )


@router.post("/local_models", summary="Upload multiple models from a ZIP archive")
async def load_multiple_models(
    file: Annotated[UploadFile, File()],
    request: Request,
    db: Annotated[AsyncSession, Depends(get_session)],
    model_name: Annotated[
        str | None, Query(description="Base name for models (optional)")
    ] = None,
) -> JSONResponse:
    """Upload a ZIP archive containing multiple model directories.

    Each top-level directory in the ZIP will be treated as a separate model.
    All extracted models will be registered in the database.

    Args:
        file: ZIP archive containing model directories
        request: HTTP request object
        model_name: Optional base name for models (used as fallback)
        db: Database session

    Returns:
        JSON response with metadata about uploaded models

    Raises:
        InvalidFileError: If no file is provided or format is invalid
        HTTPException: If an error occurs during processing
    """
    if not file:
        raise InvalidFileError("No file uploaded")

    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise InvalidFileError("Only ZIP archives are supported")

    logger.debug(
        "Multi-model ZIP upload: '{}' with {} bytes",
        file.filename,
        file.size if hasattr(file, "size") else "unknown size",
    )

    result = await upload_zip_model(file, model_name, db, multi_model=True)

    model_count = result["total_models"]
    logger.info(f"Successfully uploaded {model_count} models from ZIP archive")

    models_info = [
        {
            "id": model["model_id"],
            "name": model["model_name"],
            "directory": model["model_dir"],
            "url": f"{request.url.scheme}://{request.url.netloc}{request.url.path}/{model['model_id']}",
        }
        for model in result["models"]
    ]

    headers = {}
    if result["models"]:
        first_model = result["models"][0]
        headers["Location"] = f"{request.url}/{first_model['model_id']}"

    return JSONResponse(
        content={
            "message": f"Successfully uploaded {model_count} models",
            "models": models_info,
        },
        status_code=status.HTTP_201_CREATED,
        headers=headers,
    )
