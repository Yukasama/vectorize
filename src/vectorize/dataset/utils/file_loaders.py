"""File loaders that convert raw files to pandas DataFrames."""

import json
from csv import Sniffer
from pathlib import Path
from typing import Any, Final

import chardet
import orjson
import pandas as pd
import polars as pl
from defusedxml import ElementTree
from loguru import logger

from vectorize.common.exceptions import InvalidFileError
from vectorize.config import settings
from vectorize.dataset.exceptions import InvalidXMLFormatError

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
    encoding = _detect_encoding(path)

    try:
        return pd.read_csv(
            path,
            delimiter=delim,
            encoding=encoding,
            engine="c",
            low_memory=False,
            dtype_backend="pyarrow",
        )
    except (UnicodeDecodeError, pd.errors.ParserError) as e:
        # Fallback to multiple encoding attempts
        encodings = ("utf-8-sig", "utf-8", "latin1", "cp1252")
        for enc in encodings:
            try:
                return pd.read_csv(path, delimiter=delim, encoding=enc, engine="c")
            except UnicodeDecodeError:
                logger.debug("decode fail ({}), retry", enc)
        raise InvalidFileError(f"Failed to decode CSV file {path}") from e


def _load_json(path: Path, *_: Any) -> pd.DataFrame:  # noqa: ANN401
    """Load a JSON file, supporting array or object payloads.

    Args:
        path: Path to the JSON file.
        *_: Additional arguments (ignored).

    Returns:
        pd.DataFrame: DataFrame derived from the JSON structure.
    """
    try:
        return pd.read_json(path, dtype_backend="pyarrow")
    except ValueError:
        try:
            with path.open("rb") as f:
                payload = orjson.loads(f.read())
        except ImportError:
            with path.open(encoding="utf-8") as f:
                payload = json.load(f)

        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict) and "data" in payload:
            return pd.DataFrame(payload["data"])
        return pd.json_normalize(payload)


def _load_jsonl(path: Path, *_: Any) -> pd.DataFrame:  # noqa: ANN401
    """Load a JSONL file.

    Args:
        path: Path to the JSONL file.
        *_: Additional arguments (ignored).

    Returns:
        pd.DataFrame: DataFrame with each line as a row.
    """
    try:
        df_polars = pl.read_ndjson(path)
        return df_polars.to_pandas()
    except ImportError:
        logger.info("Polars not available, using pandas fallback")
        return pd.read_json(
            path,
            lines=True,
            orient="records",
            dtype_backend="pyarrow",
            engine="pyarrow",
        )
    except Exception as e:
        logger.warning("Failed to load with polars: {}, using pandas", e)
        return pd.read_json(path, lines=True, orient="records")


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
    if root is not None and len(root) > 0:
        return pd.DataFrame([
            {child.tag: child.text or "" for child in elem} for elem in root
        ])
    raise InvalidXMLFormatError


def _load_excel(path: Path, sheet_name: int = 0) -> pd.DataFrame:
    """Read an Excel workbook sheet into a DataFrame.

    Args:
        path: Path to the XLS/XLSX file.
        sheet_name: Index or name of the sheet to load.

    Returns:
        pd.DataFrame: DataFrame with sheet contents.
    """
    try:
        engine = "openpyxl" if path.suffix.lower() == ".xlsx" else "xlrd"

        return pd.read_excel(
            path,
            sheet_name=sheet_name,
            engine=engine,
            dtype_backend="pyarrow",
        )
    except Exception:
        return pd.read_excel(path, sheet_name=sheet_name)


_load_file: Final[dict[FileFormat, Any]] = {
    FileFormat.CSV: _load_csv,
    FileFormat.JSON: _load_json,
    FileFormat.JSONL: _load_jsonl,
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
    with path.open(newline="", encoding="utf-8", errors="ignore") as csvfile:
        sample: Final = csvfile.read(1024)
        if not sample:
            return settings.default_delimiter

        delimiter_counts = {d: sample.count(d) for d in _DELIMITERS}
        most_common = max(delimiter_counts, key=lambda d: delimiter_counts[d])

        if delimiter_counts[most_common] > 0:
            return most_common

        try:
            dialect = Sniffer().sniff(sample)
            return dialect.delimiter
        except Exception:
            return settings.default_delimiter


def _detect_encoding(path: Path) -> str:
    """Detect file encoding using chardet for better accuracy."""
    try:
        with path.open("rb") as f:
            raw_data = f.read(10000)
            result = chardet.detect(raw_data)
            return result["encoding"] or "utf-8"
    except ImportError:
        return "utf-8"
