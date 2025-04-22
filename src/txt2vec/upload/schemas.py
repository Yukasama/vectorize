"""
Schemas for importing models
"""

from pydantic import BaseModel, HttpUrl


class LoadModelRequest(BaseModel):
    """Request model for loading a Hugging Face model.

    Attributes:
        model_id (str): The ID of the model to load.
        tag (str): The specific tag or version of the model to load.

    """

    model_id: str
    tag: str


class ModelRequest(BaseModel):
    """
    Request param for providing a github url
    """

    github_url: HttpUrl


# FIXME use pydantic for Model validation
