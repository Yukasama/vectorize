"""Schemas for importing models."""

from pydantic import BaseModel, Field, constr

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

    owner: constr(min_length=1) = Field(...,
        description="GitHub username or organization")
    repo_name: constr(min_length=1) = Field(...,
        description="Repository name")
    revision: str = Field("main", alias="tag",
        description="Branch or tag name (defaults to 'main')")
