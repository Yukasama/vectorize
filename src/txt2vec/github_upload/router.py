"""
Router for importing github models
"""

from loguru import logger
from fastapi import APIRouter, HTTPException

from txt2vec.github_upload.service import handle_model_download
from upload.schemas import ModelRequest

router = APIRouter()


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
