"""Paged Response Model."""

from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class PagedResponse[T](BaseModel):
    """Paged response class."""

    page: int
    size: int
    totalpages: int
    items: Sequence[T]

    @classmethod
    def from_query(
        cls, *, items: Sequence[T], page: int, size: int, total: int
    ) -> "PagedResponse[T]":
        """Factory method to create a PagedResponse from query results."""
        totalpages = (total + size - 1) // size
        return cls(
            page=page, size=size, totalpages=totalpages, total=total, items=items
        )
