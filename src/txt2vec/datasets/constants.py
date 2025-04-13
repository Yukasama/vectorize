"""For classifying datasets and file format handling."""

from enum import StrEnum

__all__ = ["DatasetType", "FileFormat"]


class DatasetType(StrEnum):
    """Types of datasets based on their structure."""

    SENTENCE_DUPLES = "sentence_duples"
    SENTENCE_TRIPLES = "sentence_triples"
    TRIPLET_DATASET = "triplet_dataset"
    UNKNOWN = "unknown"


class FileFormat(StrEnum):
    """Supported file formats."""

    CSV = "csv"
    JSON = "json"
    XML = "xml"
    EXCEL = "xlsx"
    EXCEL_LEGACY = "xls"
