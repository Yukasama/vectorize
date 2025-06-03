"""Classification of a dataset."""

from enum import StrEnum

__all__ = ["Classification"]


class Classification(StrEnum):
    """Dataset shape: two sentences (duples) or three (triples)."""

    SENTENCE_DUPLES = "D"
    SENTENCE_TRIPLES = "T"
