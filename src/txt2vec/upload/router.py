"""
Router module for handling model upload requests.

This module provides an endpoint to load Hugging Face models using a specified model ID and tag.
"""

from loguru import logger

from fastapi import APIRouter, HTTPException, Response, status, Request
from pydantic import BaseModel
from txt2vec.upload.model_service import load_model_with_tag

router = APIRouter(tags=["Model Upload"])


class LoadModelRequest(BaseModel):
    """
    Request model for loading a Hugging Face model.

    Attributes:
        model_id (str): The ID of the model to load.
        tag (str): The specific tag or version of the model to load.
    """

    model_id: str
    tag: str


@router.post("/load")
def load_model(request: LoadModelRequest, http_request: Request):
    """
    Load a model from Hugging Face using a specified model ID and tag.

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
        return Response(
            status_code=status.HTTP_201_CREATED,
            headers={"Location": f"{http_request.url}/{request.model_id}"},
            content=f"Model '{request.model_id}' with tag '{request.tag}' loaded successfully.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
