"""Options for uploading datasets."""

from pydantic import BaseModel, Field


class DatasetUploadOptions(BaseModel):
    """Options for dataset upload."""

    prompt_name: str | None = Field(
        default=None, description="Column name for the prompt"
    )
    chosen_name: str | None = Field(
        default=None, description="Column name for the chosen example or answer"
    )
    rejected_name: str | None = Field(
        default=None,
        description="Column name for the rejected example or random sentence",
    )
    sheet_index: int = Field(default=0, description="Sheet index for Excel files")
