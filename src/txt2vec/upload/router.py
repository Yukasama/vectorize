"""Router module for handling model upload requests.

This module provides an endpoint to load Hugging Face models using a specified model ID and tag.
"""

from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request, Response, status
from loguru import logger
from pydantic import BaseModel
from upload.schemas import LoadModelRequest, ModelRequest

from txt2vec.upload.model_service import handle_model_download, load_model_with_tag

router = APIRouter(tags=["Model Upload"])


@router.post("/load")
def load_model(request: LoadModelRequest, http_request: Request):
    """Load a model from Hugging Face using a specified model ID and tag.

    :param request: The request body containing the model ID and tag.
    :param http_request: The HTTP request object.
    :return: A Response object with status 201 Created and a success message.
    :raises HTTPException: If an error occurs during model loading.
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
async def add_model(request: ModelRequest):
    """
    Endpoint to download and register a model from a specified GitHub URL.

    This endpoint accepts a POST request containing a GitHub repository URL.
    It then attempts to download the model files and prepare them for use.

    Args:
        request (ModelRequest): A request body containing the GitHub URL of the model repository.

    Returns:
    JSONResponse: A response indicating success or failure,
    typically with model info or an error message.

    Raises:
        HTTPException:
            - 400 if the GitHub URL is invalid.
            - 500 if an unexpected error occurs during processing.
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
