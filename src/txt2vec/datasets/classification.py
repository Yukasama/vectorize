"""Classification types for datasets."""

from enum import StrEnum


class Classification(StrEnum):
    """Types of datasets based on their structure."""

    SENTENCE_DUPLES = "D"
    SENTENCE_TRIPLES = "T"
