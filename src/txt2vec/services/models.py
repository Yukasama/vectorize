from enum import Enum


class DatasetType(str, Enum):
    """Types of datasets based on their structure"""

    SENTENCE_DUPLES = "sentence_duples"
    SENTENCE_TRIPLES = "sentence_triples"
    TRIPLET_DATASET = "triplet_dataset"
    UNKNOWN = "unknown"


class FileFormat(str, Enum):
    """Supported file formats"""

    CSV = "csv"
    JSON = "json"
    XML = "xml"
    EXCEL = "xlsx"
    EXCEL_LEGACY = "xls"
