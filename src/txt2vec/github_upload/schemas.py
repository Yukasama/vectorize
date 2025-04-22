"""
Schemas for importing GitHub models
"""

from pydantic import BaseModel, HttpUrl


class ModelRequest(BaseModel):
    """
    Request param for providing a github url
    """

    github_url: HttpUrl


# FIXME use pydantic?
