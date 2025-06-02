"""Schemas for importing models."""

from typing import Annotated

from pydantic import BaseModel, Field

__all__ = ["GitHubModelRequest", "HuggingFaceModelRequest"]


class HuggingFaceModelRequest(BaseModel):
    """Request model for loading a Hugging Face model.

    Attributes:
        model_tag (str): The tag of the model to load.
        revision (Optional[str]): The specific revision or version of the model to load.
        Defaults to "main".
    """

    model_tag: str
    revision: str = "main"


class GitHubModelRequest(BaseModel):
    """Request model for specifying GitHub repo access via components."""
    owner: Annotated[str, Field(min_length=1,
        description="GitHub username or organization")]
    repo_name: Annotated[str, Field(min_length=1,
        description="Repository name")]
    revision: Annotated[str, Field("main", alias="tag",
        description="Branch or tag name (defaults to 'main')")]
