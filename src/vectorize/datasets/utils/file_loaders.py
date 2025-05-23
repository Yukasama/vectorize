"""File loaders that convert raw files to pandas DataFrames."""

import json
from csv import Sniffer
from pathlib import Path
from typing import Any, Final

import pandas as pd
from defusedxml import ElementTree
from loguru import logger

from vectorize.config import settings

from ..file_format import FileFormat

__all__ = ["_load_file"]


_DELIMITERS: Final[tuple[str, ...]] = (",", ";", "\t", "|")


def _load_csv(path: Path, *_: Any) -> pd.DataFrame:  # noqa: ANN401
    """Load a CSV file with delimiter auto detection and encoding fallbacks.

    Args:
        path: Absolute path to the CSV file on disk.
        *_: Additional arguments (ignored).

    Returns:
        pd.DataFrame: Parsed DataFrame.

    Raises:
        UnicodeDecodeError: If all encoding attempts fail.
    """
    delim = _detect_delimiter(path)
    encodings = ("utf-8-sig", "utf-8", "latin1", "cp1252")
    for enc in encodings:
        try:
            return pd.read_csv(path, delimiter=delim, encoding=enc)
        except UnicodeDecodeError:
            logger.debug("decode fail ({}), retry", enc)

    # Last attempt with engine="python" to tolerate malformed lines.
    return pd.read_csv(
        path,
        delimiter=delim,
        encoding="latin1",
        engine="python",
        on_bad_lines="skip",
    )


def _load_json(path: Path, *_: Any) -> pd.DataFrame:  # noqa: ANN401
    """Load a JSON file, supporting array or object payloads.

    Args:
        path: Path to the JSON file.
        *_: Additional arguments (ignored).

    Returns:
        pd.DataFrame: DataFrame derived from the JSON structure.
    """
    try:
        return pd.read_json(path)
    except ValueError:
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict) and "data" in payload:
            return pd.DataFrame(payload["data"])
        return pd.json_normalize(payload)


def _load_xml(path: Path, *_: Any) -> pd.DataFrame:  # noqa: ANN401
    """Parse an XML document into a flat DataFrame of child elements.

    Args:
        path: Path to the XML file.
        *_: Additional arguments (ignored).

    Returns:
        pd.DataFrame: DataFrame where each XML element becomes one record.
    """
    tree = ElementTree.parse(path)
    root = tree.getroot()
    return pd.DataFrame([{child.tag: child.text for child in elem} for elem in root])


def _load_excel(path: Path, sheet_name: int = 0) -> pd.DataFrame:
    """Read an Excel workbook sheet into a DataFrame.

    Args:
        path: Path to the XLS/XLSX file.
        sheet_name: Index or name of the sheet to load.

    Returns:
        pd.DataFrame: DataFrame with sheet contents.
    """
    return pd.read_excel(path, sheet_name=sheet_name)


_load_file: Final[dict[FileFormat, Any]] = {
    FileFormat.CSV: _load_csv,
    FileFormat.JSON: _load_json,
    FileFormat.XML: _load_xml,
    FileFormat.EXCEL: _load_excel,
    FileFormat.EXCEL_LEGACY: _load_excel,
}


# -----------------------------------------------------------------------------
# Utility ---------------------------------------------------------------------
# -----------------------------------------------------------------------------


def _detect_delimiter(path: Path) -> str:
    """Detect the delimiter used in a CSV file.

    Args:
        path: Path to the CSV file.

    Returns:
        str: Detected delimiter character.
    """
    with path.open(newline="", encoding="utf-8") as csvfile:
        sample: Final = csvfile.read(4096)
        if not sample:
            return settings.default_delimiter
        try:
            dialect = Sniffer().sniff(sample)
            return dialect.delimiter
        except Exception:
            for d in _DELIMITERS:
                if d in sample:
                    return d
            return settings.default_delimiter
