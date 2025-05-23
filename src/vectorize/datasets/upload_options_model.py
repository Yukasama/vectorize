"""Options for uploading datasets."""

from pydantic import BaseModel, Field


class DatasetUploadOptions(BaseModel):
    """Options for dataset upload."""

    question_name: str | None = Field(
        default=None, description="Column name for the question"
    )
    positive_name: str | None = Field(
        default=None, description="Column name for the positive example or answer"
    )
    negative_name: str | None = Field(
        default=None,
        description="Column name for the negative example or random sentence",
    )
    sheet_index: int = Field(default=0, description="Sheet index for Excel files")
