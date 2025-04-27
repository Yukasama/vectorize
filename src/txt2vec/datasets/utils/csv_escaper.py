"""Escape CSV formulas and validate CSV format in a DataFrame."""

import string

import pandas as pd

from txt2vec.datasets.exceptions import InvalidCSVFormatError

__all__ = ["escape_csv_formulas"]

_CTL_CHARS = {chr(i) for i in range(32)}
_WS_CHARS = set(string.whitespace)


def _strip_leading_ws_ctl(value: str) -> str:
    """Remove leading whitespace & ASCII control chars without regex."""
    idx = 0
    while idx < len(value) and (value[idx] in _WS_CHARS or value[idx] in _CTL_CHARS):
        idx += 1
    return value[idx:]


def escape_csv_formulas(df: pd.DataFrame) -> pd.DataFrame:
    """Prefix dangerous strings with `'` and validate CSV format.

    Args:
        df: DataFrame to process

    Returns:
        Processed DataFrame with formula escaping

    Raises:
        InvalidCSVFormatError: When malformed CSV data is detected
    """
    _validate_csv_format(df)

    def needs_escape(val: str) -> bool:
        val = _strip_leading_ws_ctl(val)
        return val.startswith(("=", "+", "-", "@"))

    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].map(
                lambda x: f"'{x}" if isinstance(x, str) and needs_escape(x) else x
            )

    return df


def _validate_csv_format(df: pd.DataFrame) -> None:
    """Validate proper CSV formatting of DataFrame values.

    Args:
        df: DataFrame to validate

    Raises:
        InvalidCSVFormatError: When malformed CSV data is detected
    """
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            for value in df[col].dropna():
                if not isinstance(value, str):
                    continue

                quote_count = value.count('"')
                if quote_count > 0 and quote_count % 2 != 0:
                    raise InvalidCSVFormatError(f"CSV is malformed near '{value[:50]}'")

                if '""' in value and not (
                    value.startswith('"') and value.endswith('"')
                ):
                    raise InvalidCSVFormatError

                if "\n" in value or (
                    "\r" in value
                    and not (value.startswith('"') and value.endswith('"'))
                ):
                    raise InvalidCSVFormatError

                # Check for special characters that should be quoted but aren't
                if any(c in value for c in ",;|") and not (
                    value.startswith('"') and value.endswith('"')
                ):
                    raise InvalidCSVFormatError
