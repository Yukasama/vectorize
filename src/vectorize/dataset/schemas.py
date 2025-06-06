"""Schemas for dataset-related requests and responses."""

from pydantic import BaseModel, Field

__all__ = ["DatasetUploadOptions", "HuggingFaceDatasetRequest"]


class HuggingFaceDatasetRequest(BaseModel):
    """Request model for Hugging Face dataset upload."""

    dataset_tag: str = Field(
        min_length=1,
        max_length=200,
        description="Hugging Face dataset tag (e.g., 'Intel/orca_dpo_pairs')",
        examples=[
            "Intel/orca_dpo_pairs",
            "argilla/ultrafeedback-binarized-preferences",
            "HuggingFaceH4/ultrafeedback_binarized",
        ],
    )


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
