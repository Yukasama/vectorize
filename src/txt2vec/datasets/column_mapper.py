"""Dataset column mapper."""

from typing import TypedDict


class ColumnMapping(TypedDict, total=False):
    """Mapping of columns in the dataset."""

    question: str | None
    positive: str | None
    negative: str | None
