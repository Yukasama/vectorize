"""Classification utilities and strict column validation."""

from collections.abc import Mapping
from enum import StrEnum

import pandas as pd

from .column_mapper import ColumnMapping
from .exceptions import InvalidCSVColumnError, InvalidCSVFormatError

__all__ = ["Classification", "classify_dataset"]


class Classification(StrEnum):
    """Dataset shape: two sentences (duples) or three (triples)."""

    SENTENCE_DUPLES = "D"
    SENTENCE_TRIPLES = "T"


_ALIASES: Mapping[str, tuple[str, ...]] = {
    "question": ("anchor", "q", "query"),
    "positive": ("answer",),
    "negative": ("random",),
}

_ROLES = ("question", "positive", "negative")


def classify_dataset(
    df: pd.DataFrame,
    mapping: ColumnMapping | None = None,
) -> tuple[pd.DataFrame, Classification]:
    """Validate, clean, and classify a sentence dataset.

    Args:
        df: Input DataFrame containing the dataset to classify.
        mapping: Optional column mapping to identify specific fields.
            If None, default column names and aliases will be used.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Cleaned DataFrame with standardized column names
            - Classification: Enum indicating SENTENCE_DUPLES or SENTENCE_TRIPLES

    Raises:
        InvalidCSVColumnError: When a column name specified in mapping is
            not found in the DataFrame.
        InvalidCSVFormatError: When mandatory columns cannot be resolved
            using defaults or aliases.
    """
    df_cols_lc = {c.lower(): c for c in df.columns}
    header_map = _resolve_headers(df_cols_lc, mapping)

    ordered_roles = ["question", "positive"] + (
        ["negative"] if "negative" in header_map else []
    )
    cleaned = df[[header_map[r] for r in ordered_roles]].copy()
    cleaned.columns = ordered_roles

    cls = (
        Classification.SENTENCE_TRIPLES
        if "negative" in header_map
        else Classification.SENTENCE_DUPLES
    )
    return cleaned, cls


def _resolve_headers(
    df_cols_lc: dict[str, str],
    mapping: ColumnMapping | None = None,
) -> dict[str, str]:
    """Resolve dataset column headers to standardized role names.

    Args:
        df_cols_lc: Dictionary mapping lowercase column names to their
            original case-sensitive names.
        mapping: Optional mapping that specifies which columns correspond
            to which roles. If None, uses default column names and aliases.

    Returns:
        dict: Mapping from role names to actual column names in the DataFrame.

    Raises:
        InvalidCSVColumnError: When a column name specified in mapping is
            not found in the DataFrame.
        InvalidCSVFormatError: When mandatory columns (question, positive)
            cannot be resolved.
    """
    resolved: dict[str, str] = {}

    if mapping:
        for role, explicit in mapping.items():
            if explicit is None:
                continue
            col_lc = explicit.lower()
            if col_lc not in df_cols_lc:
                raise InvalidCSVColumnError(explicit)
            resolved[role] = df_cols_lc[col_lc]

    for role in _ROLES:
        if role in resolved:
            continue
        candidates = (role, *_ALIASES.get(role, ()))
        for cand in candidates:
            if cand.lower() in df_cols_lc:
                resolved[role] = df_cols_lc[cand.lower()]
                break

    if "question" not in resolved:
        raise InvalidCSVFormatError("Missing column: question")
    if "positive" not in resolved:
        raise InvalidCSVFormatError("Missing column: positive")
    if mapping and mapping.get("negative") is not None and "negative" not in resolved:
        raise InvalidCSVColumnError(mapping["negative"])
    return resolved
