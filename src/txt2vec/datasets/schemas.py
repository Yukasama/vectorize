"""Response types for datasets."""

from pydantic import BaseModel

__all__ = ["DatasetUploadResponse"]


class DatasetUploadResponse(BaseModel):
    """Response model for dataset operations."""

    filename: str
    rows: int
    columns: list[str]
    dataset_type: str
