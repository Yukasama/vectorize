"""EmbeddingRequest model cloning OpenAI API Endpoint."""

from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from vectorize.config.errors import ErrorNames

__all__ = ["EmbeddingRequest"]


class EmbeddingRequest(BaseModel):
    """Request body for Embeddings Endpoint, modelled after OpenAI's spec."""

    input: str | list[str] | list[list[int]] | list[int] = Field(
        description=(
            "Input text or tokens to embed. Accepts a single string, list of strings, "
            "or a list of token arrays."
        ),
    )

    model: str = Field(
        description="ID of the model to use, e.g. `text-embedding-ada-002`."
    )

    dimensions: int | None = Field(
        None,
        description=(
            "Optional number of dimensions for resulting embeddings (text-embedding-3)."
        ),
        ge=1,
    )

    encoding_format: str = Field(
        default="float",
        description="Either `float` (default) or `base64` for the embedding values.",
    )

    user: str | None = Field(
        None,
        description="An identifier for your end-user (helps OpenAI monitor for abuse).",
    )

    @field_validator("encoding_format")
    @classmethod
    def validate_encoding_format(cls, v: str) -> Literal["float", "base64"]:
        """Validate that encoding_format is either 'float' or 'base64'."""
        if v not in {"float", "base64"}:
            raise ValidationError(ErrorNames.ENCODING_FORMAT_ERROR)
        return v
