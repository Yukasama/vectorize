"""Normalize a DataFrame to a canonical [prompt, chosen, (rejected)] layout."""

from collections.abc import Mapping

import pandas as pd

from ..exceptions import MissingColumnError

__all__ = ["_normalize_dataset"]


_ALIASES: Mapping[str, tuple[str, ...]] = {
    "prompt": ("anchor", "question", "q", "query"),
    "chosen": ("answer", "positive", "a"),
    "rejected": ("random", "negative", "no_context"),
}

_ROLES = ("prompt", "chosen", "rejected")


def _normalize_dataset(
    df: pd.DataFrame, mapping: Mapping[str, str] | None = None
) -> None:
    """Normalize DataFrame columns to canonical names and order.

    Mutates the provided DataFrame in-place to standardize column names to
    'prompt', 'chosen', and optionally 'rejected', and reorders columns
    in that sequence. Columns not matching these roles are removed.

    Args:
        df: DataFrame to normalize (modified in-place).
        mapping: Optional explicit mapping {role: col_name} to override
            automatic column detection. If provided, these names must exist
            in the DataFrame.

    Raises:
        InvalidCSVFormatError: If required columns (prompt, chosen) cannot
            be found in the DataFrame.
    """
    header_map = _build_header_map(df, mapping)

    keep_columns = set(header_map.values())
    df.drop(columns=[c for c in df.columns if c not in keep_columns], inplace=True)
    df.rename(columns={header_map[r]: r for r in header_map}, inplace=True)

    cols_order = ["prompt", "chosen"]
    if "rejected" in header_map:
        cols_order.append("rejected")
    df[:] = df[cols_order]


def _find_column_match(
    role: str, df_cols_lc: dict[str, str], mapping: Mapping[str, str] | None
) -> str | None:
    """Find matching column for a semantic role.

    Args:
        role: Semantic role to find ('prompt', 'chosen', or 'rejected').
        df_cols_lc: Dictionary mapping lowercase column names to actual column names.
        mapping: Optional explicit mapping from roles to column names.

    Returns:
        Actual column name from DataFrame if found, None otherwise.

    Raises:
        InvalidCSVFormatError: If mapping specifies a column that doesn't exist.
    """
    if mapping and mapping.get(role):
        key = mapping[role].lower()
        if key in df_cols_lc:
            return df_cols_lc[key]
        raise MissingColumnError(mapping[role])

    candidates = [role, *list(_ALIASES[role])]
    for candidate in candidates:
        if candidate.lower() in df_cols_lc:
            return df_cols_lc[candidate.lower()]


def _build_header_map(
    df: pd.DataFrame, mapping: Mapping[str, str] | None
) -> dict[str, str]:
    """Build mapping from semantic roles to actual DataFrame column names.

    Args:
        df: Source DataFrame containing the columns to map.
        mapping: Optional explicit mapping from roles to column names.

    Returns:
        Dictionary mapping role names to actual DataFrame column names.

    Raises:
        InvalidCSVFormatError: If mandatory columns (prompt, chosen)
            cannot be found in the DataFrame.
    """
    df_cols_lc = {col.lower(): col for col in df.columns}
    header_map = {}

    for role in _ROLES:
        column = _find_column_match(role, df_cols_lc, mapping)
        if column:
            header_map[role] = column

    missing_column = None
    if "prompt" not in header_map:
        missing_column = "prompt"
        raise MissingColumnError(missing_column)
    if "chosen" not in header_map:
        missing_column = "chosen"
        raise MissingColumnError(missing_column)

    return header_map
