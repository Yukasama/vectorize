"""File formats for datasets."""

from enum import StrEnum

__all__ = ["FileFormat"]


class FileFormat(StrEnum):
    """Supported file formats for datasets."""

    # No abbreviation to preserve extension names
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    XML = "xml"
    EXCEL = "xlsx"
    EXCEL_LEGACY = "xls"
