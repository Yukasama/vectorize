"""Normalize a DataFrame to a canonical [question, positive, (negative)] layout."""

from collections.abc import Mapping

import pandas as pd

from ..exceptions import InvalidCSVFormatError

__all__ = ["normalize_dataset"]


_ALIASES: Mapping[str, tuple[str, ...]] = {
    "question": ("anchor", "q", "query"),
    "positive": ("answer",),
    "negative": ("random",),
}

_ROLES = ("question", "positive", "negative")


def normalize_dataset(
    df: pd.DataFrame, mapping: Mapping[str, str] | None = None
) -> None:
    """Mutate `df` to the canonical names, and ordered [question, positive, (negative)].

    Args:
        df: the DataFrame to normalize (in-place).
        mapping: optional explicit mapping {role: col_name}. If provided,
            those names are used verbatim (and must exist on df); otherwise
            aliases+defaults are applied.

    Raises:
        InvalidCSVFormatError: if mandatory columns are not found.
    """
    header_map = _build_header_map(df, mapping)

    keep_columns = set(header_map.values())
    df.drop(columns=[c for c in df.columns if c not in keep_columns], inplace=True)
    df.rename(columns={header_map[r]: r for r in header_map}, inplace=True)

    cols_order = ["question", "positive"]
    if "negative" in header_map:
        cols_order.append("negative")
    df[:] = df[cols_order]


def _find_column_match(
    role: str, df_cols_lc: dict[str, str], mapping: Mapping[str, str] | None
) -> str | None:
    """Find matching column for a role, using either mapping or aliases."""
    if mapping and mapping.get(role):
        key = mapping[role].lower()
        if key in df_cols_lc:
            return df_cols_lc[key]
        raise InvalidCSVFormatError(f"Missing column: {mapping[role]}")

    candidates = [role, *list(_ALIASES[role])]
    for candidate in candidates:
        if candidate.lower() in df_cols_lc:
            return df_cols_lc[candidate.lower()]


def _build_header_map(
    df: pd.DataFrame, mapping: Mapping[str, str] | None
) -> dict[str, str]:
    """Build mapping from role names to actual DataFrame column names."""
    df_cols_lc = {col.lower(): col for col in df.columns}
    header_map = {}

    for role in _ROLES:
        column = _find_column_match(role, df_cols_lc, mapping)
        if column:
            header_map[role] = column

    if "question" not in header_map:
        raise InvalidCSVFormatError("Missing column: question")
    if "positive" not in header_map:
        raise InvalidCSVFormatError("Missing column: positive")

    return header_map
