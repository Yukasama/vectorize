"""Schemas for importing models."""


from pydantic import BaseModel, Field, HttpUrl


class HuggingFaceModelRequest(BaseModel):
    """Request model for loading a Hugging Face model.

    Attributes:
        model_id (str): The ID of the model to load.
        tag (Optional[str]): The specific tag or version of the model to load.
        Defaults to "main".
    """

    model_id: str
    tag: str = "main"


class GitHubModelRequest(BaseModel):
    repo_url: HttpUrl = Field(..., alias="github_url")
    revision: str | None = Field(None, alias="tag")
    class Config:
        allow_population_by_field_name = True
