"""File formats for datasets."""

from enum import StrEnum

__all__ = ["FileFormat"]


class FileFormat(StrEnum):
    """Supported file formats."""

    CSV = "csv"
    JSON = "json"
    XML = "xml"
    EXCEL = "xlsx"
    EXCEL_LEGACY = "xls"
