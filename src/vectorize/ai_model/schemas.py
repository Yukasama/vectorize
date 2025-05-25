"""Paged Response Model."""
from typing import TypeVar

from pydantic.generics import GenericModel

T = TypeVar('T')


class PagedResponse[T](GenericModel):
    """Paged response class.

    Args:
        GenericModel (_type_): _description_
        Generic (_type_): _description_

    Returns:
        _type_: _description_
    """
    page: int
    size: int
    totalpages: int
    items: list[T]

    @classmethod
    def from_query(cls, *, items: list[T],
                    page: int, size: int, total: int) -> "PagedResponse[T]":
        """Ctor.

        Args:
            items (list[T]): _description_
            page (int): _description_
            size (int): _description_
            total (int): _description_

        Returns:
            PagedResponse[T]: _description_
        """
        pages = (total + size - 1) // size
        return cls(page=page, size=size, total=total, pages=pages, items=items)
