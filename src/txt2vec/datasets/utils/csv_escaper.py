"""Escape CSV formulas and validate CSV format in a DataFrame."""

import string

import pandas as pd

from txt2vec.datasets.exceptions import InvalidCSVFormatError

__all__ = ["escape_csv_formulas"]


_CTL_CHARS = {chr(i) for i in range(32)}
_WS_CHARS = set(string.whitespace)
_CSV_SPECIAL_CHARS = ",;|\n\r"


def escape_csv_formulas(df: pd.DataFrame) -> pd.DataFrame:
    """Prefix dangerous strings with `'` and validate CSV format.

    Args:
        df: DataFrame to process

    Returns:
        Processed DataFrame with formula escaping

    Raises:
        InvalidCSVFormatError: When malformed CSV data is detected
    """
    result = df.copy()

    for col in result.columns:
        if pd.api.types.is_string_dtype(result[col]):
            result[col] = _process_column(result[col])

    return result


def _strip_leading_ws_ctl(value: str) -> str:
    """Remove leading whitespace & ASCII control chars without regex."""
    idx = 0
    while idx < len(value) and (value[idx] in _WS_CHARS or value[idx] in _CTL_CHARS):
        idx += 1
    return value[idx:]


def _process_column(series: pd.Series) -> pd.Series:
    """Process a pandas Series to escape formulas and validate format.

    Args:
        series: DataFrame column to process

    Returns:
        Processed column with escaped formulas

    Raises:
        InvalidCSVFormatError: When malformed CSV data is detected
    """
    result = series.copy()

    for i, value in enumerate(series):
        if not isinstance(value, str):
            continue

        stripped_val = _strip_leading_ws_ctl(value)

        quote_count = value.count('"')
        if quote_count > 0 and quote_count % 2 != 0:
            raise InvalidCSVFormatError(f"CSV is malformed near '{value[:50]}'")

        if '""' in value and not (value.startswith('"') and value.endswith('"')):
            raise InvalidCSVFormatError(
                "Double-quotes within value must be properly escaped"
            )

        has_special_chars = any(c in value for c in _CSV_SPECIAL_CHARS)
        is_properly_quoted = value.startswith('"') and value.endswith('"')
        if has_special_chars and not is_properly_quoted:
            raise InvalidCSVFormatError("Special characters must be properly quoted")

        if stripped_val.startswith(("=", "+", "-", "@")):
            result.iat[i] = f"'{value}"

    return result
