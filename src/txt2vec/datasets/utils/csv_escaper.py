"""Escape CSV formulas in a DataFrame."""

import string

import pandas as pd

__all__ = ["escape_csv_formulas"]

_CTL_CHARS = {chr(i) for i in range(32)}
_WS_CHARS = set(string.whitespace)


def _strip_leading_ws_ctl(value: str) -> str:
    """Remove leading whitespace & ASCII control chars without regex."""
    idx = 0
    while idx < len(value) and (value[idx] in _WS_CHARS or value[idx] in _CTL_CHARS):
        idx += 1
    return value[idx:]


def escape_csv_formulas(df: pd.DataFrame) -> None:
    """Prefix dangerous strings with `'` so spreadsheets don't evaluate formulas."""

    def needs_escape(val: str) -> bool:
        val = _strip_leading_ws_ctl(val)
        return val.startswith(("=", "+", "-", "@"))

    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].map(
                lambda x: f"'{x}" if isinstance(x, str) and needs_escape(x) else x
            )
