"""Classification utilities and strict column validation."""

from collections.abc import Mapping
from typing import cast

import pandas as pd

from vectorize.config.errors import ErrorNames

from ..classification import Classification
from ..column_mapper import ColumnMapping
from ..exceptions import InvalidCSVColumnError, MissingColumnError

__all__ = ["_classify_dataset"]


_ALIASES: Mapping[str, tuple[str, ...]] = {
    "prompt": ("anchor", "q", "query"),
    "chosen": ("answer",),
    "rejected": ("random",),
}

_ROLES = ("prompt", "chosen", "rejected")


def _classify_dataset(
    df: pd.DataFrame, mapping: ColumnMapping | None = None
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

    ordered_roles = ["prompt", "chosen"] + (
        ["rejected"] if "rejected" in header_map else []
    )
    cleaned = df[[header_map[r] for r in ordered_roles]].copy()
    cleaned.columns = ordered_roles

    cls = (
        Classification.SENTENCE_TRIPLES
        if "rejected" in header_map
        else Classification.SENTENCE_DUPLES
    )
    return cast(pd.DataFrame, cleaned), cls


def _resolve_headers(
    df_cols_lc: dict[str, str], mapping: ColumnMapping | None = None
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
        MissingColumnError: When mandatory columns (prompt, chosen, or
            an explicitly requested rejected) cannot be resolved.
    """
    resolved: dict[str, str] = {}

    if mapping:
        for role, explicit in mapping.items():
            if explicit is None:
                continue
            if not isinstance(explicit, str):
                raise InvalidCSVColumnError(ErrorNames.INVALID_COLUMN_TYPE)
            col_lc = explicit.lower()
            if col_lc not in df_cols_lc:
                raise InvalidCSVColumnError(str(explicit))
            resolved[role] = df_cols_lc[col_lc]

    for role in _ROLES:
        if role in resolved:
            continue
        col_name = next(
            (
                df_cols_lc[c.lower()]
                for c in (role, *_ALIASES.get(role, ()))
                if c.lower() in df_cols_lc
            ),
            None,
        )
        if col_name:
            resolved[role] = col_name

    mandatory = ["prompt", "chosen"]
    if mapping and mapping.get("rejected") is not None:
        mandatory.append("rejected")

    for role in mandatory:
        if role not in resolved:
            explicit = (
                mapping[role] if mapping and role in mapping and mapping[role] else role
            )
            raise MissingColumnError(explicit)

    return resolved
