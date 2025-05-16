"""Router for model upload and management."""

from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from txt2vec.common.status import TaskStatus
from txt2vec.config.db import get_session
from txt2vec.datasets.exceptions import InvalidFileError
from txt2vec.upload.background_service import write_to_database
from txt2vec.upload.exceptions import ServiceUnavailableError
from txt2vec.upload.github_service import handle_model_download
from txt2vec.upload.huggingface_service import load_model_and_save_to_db
from txt2vec.upload.models import UploadTask
from txt2vec.upload.schemas import GitHubModelRequest, HuggingFaceModelRequest
from txt2vec.upload.zip_service import upload_zip_model

router = APIRouter(tags=["Model Upload"])


@router.post("/load", status_code=status.HTTP_201_CREATED)
async def load_model_huggingface(
    data: HuggingFaceModelRequest,
    http_request: Request,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    """Load a Hugging Face model and return a Location header.

    This endpoint loads a Hugging Face model using the provided model ID and
    tag, caches it locally, and stores it in the database. If successful, it
    returns a 201 Created response with a Location header pointing to the
    model.

    Args:
        data (HuggingFaceModelRequest): Contains the model ID and tag.
        http_request (Request): The HTTP request object.
        db (AsyncSession): The database session.

    Returns:
        A 201 Created response with a Location header.

    Raises:
        HTTPException: If an error occurs during model loading or processing.
    """
    try:
        logger.debug(f"Ladeanfrage: {data.model_id}@{data.tag}")
        await load_model_and_save_to_db(data.model_id, data.tag, db)

        key = f"{data.model_id}@{data.tag}"
        return Response(
            status_code=status.HTTP_201_CREATED,
            headers={"Location": f"{http_request.url}/{key}"},
        )
    except Exception as e:
        logger.exception("Fehler beim Laden:")
        raise ServiceUnavailableError from e


@router.post("/add_model", status_code=status.HTTP_201_CREATED)
async def load_model_github(
    request: GitHubModelRequest, db: Annotated[AsyncSession, Depends(get_session)]
) -> dict:
    """Create an upload task and register a model from a specified GitHub repository.

    This endpoint stores a new UploadTask in the database using the given
    GitHub URL and metadata. It then begins the model download process.

    Args:
        request (GitHubModelRequest): Includes the GitHub URL and model tag.
        db (AsyncSession): Async database session.

    Returns:
        dict: JSON response containing the UploadTask ID.
    """
    logger.info("Received request to add model from GitHub URL: {}", request.github_url)

    upload_task = UploadTask(
        model_tag=request.model_tag,
        source=request.source,
        task_status=TaskStatus.PENDING,
    )

    await write_to_database(db, upload_task)

    await handle_model_download(request.github_url)

    logger.info("Model handled successfully for: {}", request.github_url)

    return {"id": str(upload_task.id)}


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
