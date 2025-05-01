"""Router for model upload and management."""

import json
from typing import Annotated, Any
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

from txt2vec.errors import ErrorCode
from txt2vec.upload.exceptions import (
    EmptyModelError,
    InvalidModelError,
    InvalidZipError,
    ModelTooLargeError,
    NoValidModelsFoundError,
    UnsupportedModelFormatError,
)
from txt2vec.upload.github_service import handle_model_download
from txt2vec.upload.local_service import upload_embedding_model
from txt2vec.upload.model_service import load_model_with_tag
from txt2vec.upload.schemas import GitHubModelRequest, HuggingFaceModelRequest

router = APIRouter(tags=["Model Upload"])


@router.post("/load")
def load_huggingface_model(
    request: HuggingFaceModelRequest,
    http_request: Request,
) -> Response:
    """Load a model from Hugging Face using the specified model ID and tag.

    Args:
        request: Contains the model ID and tag.
        http_request: The HTTP request object used to build the Location header.

    Returns:
        A 201 Created response with a Location header.

    Raises:
        HTTPException: If an unexpected error occurs during model loading.

    """
    try:
        logger.debug(
            "Loading model: model_id={}, tag={}",
            request.model_id,
            request.tag,
        )
        load_model_with_tag(request.model_id, request.tag)

        safe_model_id = quote(request.model_id, safe="")
        location = f"{http_request.url}/{safe_model_id}"
        return Response(
            status_code=status.HTTP_201_CREATED,
            headers={"Location": location},
        )
    except Exception as e:
        logger.error("Error loading Hugging Face model: {}", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load model",
        ) from e


@router.post("/add_model")
async def load_github_model(request: GitHubModelRequest) -> dict[str, Any]:
    """Download and register a model from a specified GitHub repository.

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

    try:
        result = await handle_model_download(request.github_url)
        logger.info(
            "Model handled successfully for: {}",
            request.github_url,
        )
        return result
    except HTTPException:
        logger.warning(
            "Handled HTTPException for GitHub URL: {}",
            request.github_url,
        )
        raise
    except Exception as e:
        logger.exception(
            "Unhandled error during GitHub model import for URL: {}",
            request.github_url,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from e


def _create_error_response(status_code: int, code: ErrorCode, message: str) -> dict:
    """Create a standardized error response dictionary.

    Args:
        status_code: HTTP status code
        code: Application error code
        message: Error message

    Returns:
        Dictionary with code and message
    """
    return {"code": str(code), "message": message}


def _handle_model_upload_error(e: Exception) -> Response:
    """Handle different upload error types with appropriate responses.

    Args:
        e: The exception that was raised

    Returns:
        Response object with error details
    """
    if isinstance(e, EmptyModelError):
        logger.warning("Empty model file: {}", str(e))
        status_code = status.HTTP_400_BAD_REQUEST
        error_code = ErrorCode.EMPTY_FILE
    elif isinstance(e, ModelTooLargeError):
        logger.warning("Model too large: {}", str(e))
        status_code = status.HTTP_400_BAD_REQUEST
        error_code = ErrorCode.INVALID_FILE
    elif isinstance(e, UnsupportedModelFormatError):
        logger.warning("Unsupported model format: {}", str(e))
        status_code = status.HTTP_400_BAD_REQUEST
        error_code = ErrorCode.UNSUPPORTED_FORMAT
    elif isinstance(e, (InvalidZipError, NoValidModelsFoundError, InvalidModelError)):
        logger.warning("{}: {}", e.__class__.__name__, str(e))
        status_code = status.HTTP_400_BAD_REQUEST
        error_code = ErrorCode.INVALID_FILE
    else:
        logger.exception("Unhandled error during model upload")
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        error_code = ErrorCode.SERVER_ERROR
        # Use a generic message for unexpected errors
        return Response(
            status_code=status_code,
            content=json.dumps(
                _create_error_response(
                    status_code, error_code, "Error processing upload"
                )
            ),
            media_type="application/json",
        )

    # For expected errors, use the exception message
    return Response(
        status_code=status_code,
        content=json.dumps(_create_error_response(status_code, error_code, str(e))),
        media_type="application/json",
    )


@router.post("/models")
async def load_local_model(
    request: Request,
    model_name: Annotated[
        str,
        Query(description="Name for the uploaded model"),
    ],
    extract_zip: Annotated[
        bool,
        Query(description="Whether to extract ZIP files"),
    ] = True,
    files: list[UploadFile] | None = None,
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
    # Manuelle Validierung für leere Dateien
    if not files:
        logger.warning("No files provided in request")
        return Response(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=json.dumps(
                _create_error_response(
                    status.HTTP_400_BAD_REQUEST,
                    ErrorCode.INVALID_FILE,
                    "No files provided for upload",
                )
            ),
            media_type="application/json",
        )

    logger.debug(
        "Uploading model '{}' with {} files",
        model_name,
        len(files),
    )

    try:
        result = await upload_embedding_model(files, model_name, extract_zip)
        logger.info(
            "Successfully uploaded model: {}",
            result["model_dir"],
        )

        return Response(
            status_code=status.HTTP_201_CREATED,
            headers={"Location": f"{request.url}/{result['model_id']}"},
        )
    except Exception as e:
        return _handle_model_upload_error(e)
