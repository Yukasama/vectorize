"""
Schemas for importing models
"""

from pydantic import BaseModel, HttpUrl


class HuggingFaceModelRequest(BaseModel):
    """Request model for loading a Hugging Face model.

    Attributes:
        model_id (str): The ID of the model to load.
        tag (str): The specific tag or version of the model to load.

    """

    model_id: str
    tag: str


class GitHubModelRequest(BaseModel):
    """
    Request param for loading a model from GitHub

    Attributes:
        github_url (HttpUrl): The url to the model to be loaded.
    """

    github_url: HttpUrl


# FIXME use pydantic for Model validation
