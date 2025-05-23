"""Schemas for importing models."""


from pydantic import BaseModel, Field, HttpUrl


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
    """_RequestModel for passing (GitHub) Urls.

    Args:
        BaseModel (_type_): _description_
    """
    repo_url: HttpUrl = Field(..., alias="github_url")
    revision: str | None = Field(None, alias="tag")
