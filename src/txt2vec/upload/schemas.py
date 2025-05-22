"""Schemas for importing models."""


from pydantic import BaseModel, HttpUrl


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
    """Request param for loading a model from GitHub.

    Attributes:
        github_url (HttpUrl): The URL to the model to be loaded.
    """

    github_url: HttpUrl
