"""Dataset column mapper."""

from typing import TypedDict

__all__ = ["ColumnMapping"]


class ColumnMapping(TypedDict, total=False):
    """Mapping of columns in the dataset."""

    prompt: str | None
    chosen: str | None
    rejected: str | None
