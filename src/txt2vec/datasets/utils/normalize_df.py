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
    df_cols_lc = {col.lower(): col for col in df.columns}

    header_map: dict[str, str] = {}
    if mapping:
        for role in _ROLES:
            if mapping.get(role):
                key = mapping[role].lower()
                if key not in df_cols_lc:
                    raise InvalidCSVFormatError(f"Missing column: {mapping[role]}")
                header_map[role] = df_cols_lc[key]

    for role in _ROLES:
        if role in header_map:
            continue
        for candidate in (role, *_ALIASES[role]):
            if candidate.lower() in df_cols_lc:
                header_map[role] = df_cols_lc[candidate.lower()]
                break

    if "question" not in header_map:
        raise InvalidCSVFormatError("Missing column: question")
    if "positive" not in header_map:
        raise InvalidCSVFormatError("Missing column: positive")

    keep = set(header_map.values())
    drop_cols = [c for c in df.columns if c not in keep]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    df.rename(columns={header_map[r]: r for r in header_map}, inplace=True)

    cols_order = ["question", "positive"]
    if "negative" in header_map:
        cols_order.append("negative")
    df[:] = df[cols_order]
