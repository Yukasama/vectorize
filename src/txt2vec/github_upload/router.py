"""
Router for importing github models
"""

from loguru import logger
from fastapi import APIRouter, HTTPException

from txt2vec.github_upload.service import handle_model_download
from txt2vec.github_upload.schemas import ModelRequest

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
    try:
        return await handle_model_download(request.github_url)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error("GitHub model download not possible: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
