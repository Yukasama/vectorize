"""Router for model upload and management."""

from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from txt2vec.config.db import get_session
from txt2vec.datasets.exceptions import InvalidFileError
from txt2vec.upload.exceptions import ServiceUnavailableError
from txt2vec.upload.github_service import handle_model_download
from txt2vec.upload.huggingface_service import load_model_and_save_to_db
from txt2vec.upload.local_service import upload_embedding_model
from txt2vec.upload.schemas import GitHubModelRequest, HuggingFaceModelRequest

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


@router.post("/add_model")
async def load_model_github(request: GitHubModelRequest) -> Response:
    """Download and register a model from a specified GitHub repository.

    This endpoint accepts a GitHub repository URL and attempts to download
    and prepare the model files for use. If successful, a JSON response is returned.

    Args:
        request: Contains the GitHub repository URL.

    Returns:
        A response indicating success or error details.

    Raises:
        HTTPException:
            - 400 if the GitHub URL is invalid.
            - 500 if an unexpected error occurs during model processing.

    """
    logger.info(
        "Received request to add model from GitHub URL: {}",
        request.github_url,
    )

    result = await handle_model_download(request.github_url)
    logger.info(
        "Model handled successfully for: {}",
        request.github_url,
    )
    return result


@router.post("/models")
async def load_model_local(
    files: list[UploadFile],
    request: Request,
    model_name: Annotated[str, Query(description="Name for the uploaded model")],
    extract_zip: Annotated[
        bool,
        Query(description="Whether to extract ZIP files"),
    ] = True,
) -> Response:
    """Upload PyTorch model files to the server.

    Args:
        request: The HTTP request object.
        model_name: Name to assign to the uploaded model.
        extract_zip: Whether to extract ZIP files (default: True).
        files: The uploaded model files.

    Returns:
        A 201 Created response with a Location header.

    Raises:
        HTTPException: If an error occurs during file upload or processing.

    """
    if not files:
        raise InvalidFileError

    logger.debug(
        "Uploading model '{}' with {} files",
        model_name,
        len(files),
    )

    result = await upload_embedding_model(files, model_name, extract_zip)
    logger.info(
        "Successfully uploaded model: {}",
        result["model_dir"],
    )

    return Response(
        status_code=status.HTTP_201_CREATED,
        headers={"Location": f"{request.url}/{result['model_id']}"},
    )
