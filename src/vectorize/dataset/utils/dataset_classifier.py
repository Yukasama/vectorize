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
    """Resolve DataFrame headers to canonical roles.

    Args:
        df_cols_lc: Mapping of lowercase header → original header.
        mapping: Optional explicit ``ColumnMapping`` from the request payload.
            ``None`` values inside the mapping indicate that the role should
            be ignored.

    Returns:
        Dictionary mapping role names (``prompt``, ``chosen``, …) to the
        original header names in the DataFrame.

    Raises:
        InvalidCSVColumnError: If an explicit column is missing or of wrong
            type.
        MissingColumnError: If mandatory columns (``prompt``, ``chosen``, or
            an explicitly requested ``rejected``) could not be resolved.
    """
    resolved: dict[str, str] = {}

    if mapping:
        resolved.update(_apply_explicit_mapping(mapping, df_cols_lc))

    for role in _ROLES:
        if role in resolved:
            continue
        col_name = next(
            (
                df_cols_lc[c_lc]
                for c in (role, *_ALIASES.get(role, ()))
                if (c_lc := c.lower()) in df_cols_lc
            ),
            None,
        )
        if col_name:
            resolved[role] = col_name

    mandatory = ["prompt", "chosen"]
    if mapping and mapping.get("rejected") is not None:
        mandatory.append("rejected")

    missing = [r for r in mandatory if r not in resolved]
    if missing:
        role = missing[0]
        explicit = mapping.get(role) if mapping and mapping.get(role) else role
        raise MissingColumnError(str(explicit))

    return resolved


def _apply_explicit_mapping(
    mapping: ColumnMapping, df_cols_lc: dict[str, str]
) -> dict[str, str]:
    """Translate an explicit role-to-column mapping into DataFrame column names.

    Args:
        mapping: Mapping such as ``{"prompt": "my_q", "chosen": "answer"}``.
            ``None`` values mean the role should be skipped.
        df_cols_lc: Mapping of lowercase header → original header.

    Returns:
        Dictionary where keys are role names (``prompt``, ``chosen``, …) and
        values are the exact column names in the DataFrame.

    Raises:
        InvalidCSVColumnError: If a supplied column is of the wrong type or
            does not exist in ``df_cols_lc``.
    """
    resolved: dict[str, str] = {}

    for role, explicit in mapping.items():
        if explicit is None:
            continue
        if not isinstance(explicit, str):
            raise InvalidCSVColumnError(ErrorNames.INVALID_COLUMN_TYPE)

        col_lc = explicit.lower()
        if col_lc not in df_cols_lc:
            raise InvalidCSVColumnError(str(explicit))

        resolved[role] = df_cols_lc[col_lc]

    return resolved
