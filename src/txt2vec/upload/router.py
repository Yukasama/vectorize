"""Router module for handling model upload requests.

This module provides endpoints to:
1. Load Hugging Face models using a specified model ID and tag
2. Add models from GitHub repositories
3. Upload model files directly to the server
"""
from typing import Annotated, Final, List
from urllib.parse import quote

from fastapi import APIRouter, File, HTTPException, Query, Request, Response, UploadFile, status
from loguru import logger
from pydantic import BaseModel

from txt2vec.upload.github_service import handle_model_download
from txt2vec.upload.local_service import upload_embedding_model
from txt2vec.upload.model_service import load_model_HF
from txt2vec.upload.schemas import HuggingFaceModelRequest, GitHubModelRequest


router = APIRouter(tags=["Model Upload"])


@router.post("/load", status_code=status.HTTP_201_CREATED)
def load_model(request: HuggingFaceModelRequest, http_request: Request):
    """Lädt ein Modell und gibt Location-Header zurück."""
    try:
        logger.debug(f"Ladeanfrage: {request.model_id}@{request.tag}")
        load_model_HF(request.model_id, request.tag)

        key = f"{request.model_id}@{request.tag}"
        return Response(
            status_code=status.HTTP_201_CREATED,
            headers={"Location": f"{http_request.url}/{key}"},
        )
    except Exception as e:
        logger.exception("Fehler beim Laden:")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/add_model")
async def load_model_github(request: GitHubModelRequest):
    """
    Download and register a model from a specified GitHub repository.

    This endpoint accepts a GitHub repository URL and attempts to download
    and prepare the model files for use. If successful, a JSON response is returned.

    Args:
        request (GitHubModelRequest): Contains the GitHub repository URL.

    Returns:
        JSONResponse: A response indicating success or error details.

    Raises:
        HTTPException:
            - 400 if the GitHub URL is invalid.
            - 500 if an unexpected error occurs during model processing.
    """
    logger.info("Received request to add model from GitHub URL: {}", request.github_url)

    try:
        result = await handle_model_download(request.github_url)
        logger.info("Model handled successfully for: {}", request.github_url)
        return result
    except HTTPException as e:
        logger.warning(
            "Handled HTTPException for GitHub URL %s: %s", request.github_url, e.detail
        )
        raise e
    except Exception as e:
        logger.exception(
            "Unhandled error %s  during GitHub model import for URL: %s",
            e,
            request.github_url,
        )
        raise HTTPException(status_code=500, detail="Internal server error.")


@router.post("/models")
async def load_model_local(
    files: List[UploadFile],
    request: Request,
    model_name: Annotated[str, Query(description="Name for the uploaded model")],
    description: Annotated[
        str, Query(description="Description of the model")
    ] = "",
    extract_zip: Annotated[
        bool, Query(description="Whether to extract ZIP files")
    ] = True,
) -> Response:
    """
    Upload embedding model files to the server.

    This endpoint accepts multiple files representing an embedding model. If ZIP files
    are uploaded and extract_zip is True, they will be extracted before saving.
    Returns a 201 Created response with a Location header pointing to the model.

    Args:
        files (List[UploadFile]): The uploaded model files.
        request (Request): The HTTP request object.
        model_name (str): Name to assign to the uploaded model.
        description (str, optional): Optional description of the model.
        extract_zip (bool): Whether to extract ZIP files (default: True).

    Returns:
        Response: A 201 Created response with a Location header.

    Raises:
        HTTPException: If an error occurs during file upload or processing.
    """
    logger.debug("Uploading model '{}' with {} files", model_name, len(files))

    result = await upload_embedding_model(files, model_name, description, extract_zip)

    logger.info("Successfully uploaded model: {}", result["model_dir"])

    return Response(
        status_code=status.HTTP_201_CREATED,
        headers={"Location": f"{request.url}/{result['model_id']}"},
    )
