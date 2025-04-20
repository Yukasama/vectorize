from pydantic import BaseModel


class ModelRequest(BaseModel):
    github_url: str
