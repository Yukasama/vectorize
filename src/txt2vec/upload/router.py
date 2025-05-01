"""Router module for handling model upload requests.

This module provides endpoints to:
1. Load Hugging Face models using a specified model ID and tag
2. Add models from GitHub repositories
3. Upload model files directly to the server
"""
from typing import Annotated
from urllib.parse import quote

from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from loguru import logger

from txt2vec.upload.github_service import handle_model_download
from txt2vec.upload.local_service import upload_embedding_model
from txt2vec.upload.model_service import load_model_with_tag
from txt2vec.upload.schemas import GitHubModelRequest, HuggingFaceModelRequest

router = APIRouter(tags=["Model Upload"])


@router.post("/load")
def load_model_huggingface(request: HuggingFaceModelRequest, http_request: Request):
    """Load a model from Hugging Face using the specified model ID and tag.

    This endpoint loads a model based on the provided Hugging Face model ID and an optional tag.
    On success, returns a 201 Created response with a Location header pointing to the model.

    Args:
        request (HuggingFaceModelRequest): Contains the model ID and tag.
        http_request (Request): The HTTP request object used to build the Location header.

    Returns:
        Response: A 201 Created response with a Location header.

    Raises:
        HTTPException: If an unexpected error occurs during model loading.
    """
    try:
        logger.debug(
            "Loading model: model_id={}, tag={}", request.model_id, request.tag
        )
        load_model_with_tag(request.model_id, request.tag)

        safe_model_id = quote(request.model_id, safe="")
        return Response(
            status_code=status.HTTP_201_CREATED,
            headers={"Location": f"{http_request.url}/{safe_model_id}"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/add_model")
async def load_model_github(request: GitHubModelRequest):
    """Download and register a model from a specified GitHub repository.

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
    files: list[UploadFile],
    request: Request,
    model_name: Annotated[str, Query(description="Name for the uploaded model")],
    description: Annotated[
        str, Query(description="Description of the model")
    ] = "",
    extract_zip: Annotated[
        bool, Query(description="Whether to extract ZIP files")
    ] = True,
) -> Response:
    """Upload embedding model files to the server.

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
