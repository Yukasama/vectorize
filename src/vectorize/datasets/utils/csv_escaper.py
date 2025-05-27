"""Escape CSV formulas and validate CSV format in a DataFrame."""

import string

import pandas as pd

from vectorize.config.errors import ErrorNames

from ..exceptions import InvalidCSVFormatError

__all__ = ["_escape_csv_formulas"]


_CTL_CHARS = {chr(i) for i in range(32)}
_WS_CHARS = set(string.whitespace)
_CSV_SPECIAL_CHARS = ",;|\n\r"


def _escape_csv_formulas(df: pd.DataFrame) -> pd.DataFrame:
    """Prefix dangerous strings with `'` to prevent CSV formula injection attacks.

    Adds a single quote prefix to any cell that begins with =, +, -, or @ characters
    to prevent formula injection attacks in spreadsheet applications.
    Also validates CSV format integrity.

    Args:
        df: DataFrame to protect against formula injection

    Returns:
        Processed DataFrame with formula escaping

    Raises:
        InvalidCSVFormatError: When malformed CSV data is detected (unbalanced quotes,
        improper escaping, or other malicious formatting)
    """
    result = df.copy()

    for col in result.columns:
        if pd.api.types.is_string_dtype(result[col]):
            column_series: pd.Series = result[col]
            _process_column(column_series)

            for i, value in enumerate(result[col]):
                if isinstance(value, str):
                    stripped_val = _strip_leading_ws_ctl(value)
                    if stripped_val.startswith(("=", "+", "-", "@")):
                        result[col].iat[i] = f"'{value}"

    return result


def _strip_leading_ws_ctl(value: str) -> str:
    """Remove leading whitespace & ASCII control chars without regex.

    Args:
        value: String to process

    Returns:
        String with leading whitespace and control characters removed
    """
    idx = 0
    while idx < len(value) and (value[idx] in _WS_CHARS or value[idx] in _CTL_CHARS):
        idx += 1
    return value[idx:]


def _process_column(series: pd.Series) -> None:
    """Validate CSV format in a pandas Series.

    Checks each value in the series for CSV format integrity, including:
    - Balanced double-quotes
    - Properly escaped internal double-quotes
    - Correct quoting of values containing special characters

    Args:
        series: Pandas Series to validate

    Raises:
        InvalidCSVFormatError: When CSV format validation fails due to
            malformed or potentially malicious content
    """
    for value in series:
        if not isinstance(value, str):
            continue

        quote_count = value.count('"')
        if quote_count > 0 and quote_count % 2 != 0:
            raise InvalidCSVFormatError(
                ErrorNames.FORMAT_INVALID_CSV_ERROR.format(value=value[:50])
            )

        if '""' in value and not (value.startswith('"') and value.endswith('"')):
            raise InvalidCSVFormatError(ErrorNames.DOUBLE_QUOTES_ERROR)

        has_special_chars = any(c in value for c in _CSV_SPECIAL_CHARS)
        is_properly_quoted = value.startswith('"') and value.endswith('"')
        if has_special_chars and not is_properly_quoted:
            raise InvalidCSVFormatError(ErrorNames.SPECIAL_CHARS_ERROR)
