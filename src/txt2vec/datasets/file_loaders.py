"""File loaders that convert raw files to pandas DataFrames."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, Literal

import pandas as pd
from defusedxml import ElementTree
from loguru import logger

from txt2vec.config.config import app_config
from txt2vec.datasets.file_format import FileFormat

__all__ = ["FILE_LOADERS"]

# -----------------------------------------------------------------------------
# Config ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

dataset_config = app_config["dataset"]
default_delimiter = dataset_config.get("default_delimiter", ",")

Delimiter = Literal[",", ";", "\t", "|"]

# -----------------------------------------------------------------------------
# Loader functions ------------------------------------------------------------
# -----------------------------------------------------------------------------


def _load_csv(path: Path, *_: Any) -> pd.DataFrame:
    """Load a CSV file with delimiter auto detection and encoding fallbacks.

    :param path: Absolute path to the CSV file on disk.
    :returns: Parsed DataFrame.
    :raises UnicodeDecodeError: If all encoding attempts fail.
    """
    delim = _detect_delimiter(path)
    encodings = ("utf-8", "latin1", "cp1252")
    for enc in encodings:
        try:
            return pd.read_csv(path, delimiter=delim, encoding=enc)
        except UnicodeDecodeError:
            logger.debug("decode fail (%s), retry", enc)

    # Last attempt with engine="python" to tolerate malformed lines.
    return pd.read_csv(
        path,
        delimiter=delim,
        encoding="latin1",
        engine="python",
        on_bad_lines="skip",
    )


def _load_json(path: Path, *_: Any) -> pd.DataFrame:
    """Load a JSON file, supporting array or object payloads.

    :param path: Path to the JSON file.
    :returns: DataFrame derived from the JSON structure.
    :raises ValueError: When JSON cannot be decoded.
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


def _load_xml(path: Path, *_: Any) -> pd.DataFrame:
    """Parse an XML document into a flat DataFrame of child elements.

    :param path: Path to the XML file.
    :returns: DataFrame where each XML element becomes one record.
    """
    tree = ElementTree.parse(path)
    root = tree.getroot()
    return pd.DataFrame([{child.tag: child.text for child in elem} for elem in root])


def _load_excel(path: Path, sheet_name: int = 0) -> pd.DataFrame:
    """Read an Excel workbook sheet into a DataFrame.

    :param path: Path to the XLS/XLSX file.
    :param sheet_name: Index or name of the sheet to load.
    :returns: DataFrame with sheet contents.
    """
    return pd.read_excel(path, sheet_name=sheet_name)


FILE_LOADERS: Final[dict[FileFormat, Any]] = {
    FileFormat.CSV: _load_csv,
    FileFormat.JSON: _load_json,
    FileFormat.XML: _load_xml,
    FileFormat.EXCEL: _load_excel,
    FileFormat.EXCEL_LEGACY: _load_excel,
}

# -----------------------------------------------------------------------------
# Utility ---------------------------------------------------------------------
# -----------------------------------------------------------------------------


def _detect_delimiter(path: Path) -> Delimiter:
    """Guess the delimiter in a CSV sample, defaulting to config value.

    :param path: Path to the candidate CSV file.
    :returns: Detected delimiter character.
    """
    sample = path.read_text(encoding="utf-8", errors="ignore")[:4096]
    if not sample:
        return default_delimiter
    for d in (",", ";", "\t", "|"):
        if d in sample:
            return d
    return default_delimiter
