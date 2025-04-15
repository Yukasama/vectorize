"""Dataset service."""

import json
import os
import shutil
import tempfile
from csv import Sniffer
from pathlib import Path
from typing import Any, Final, Literal

import pandas as pd
from fastapi import UploadFile
from loguru import logger

from txt2vec.config import UPLOAD_DIR
from txt2vec.datasets.classification import Classification
from txt2vec.datasets.exceptions import (
    EmptyCSVError,
    InvalidCSVFormatError,
    InvalidFileError,
    UnsupportedFormatError,
)
from txt2vec.datasets.file_format import FileFormat

__all__ = ["upload_file"]

MINIMUM_COLUMNS: Final = 2
MAXIMUM_COLUMNS: Final = 3
DEFAULT_DELIMITER: Final = ";"


async def upload_file(file: UploadFile, sheet_name: int = 0) -> dict[str, Any]:
    """Process file upload from FastAPI endpoint.

    This method handles temporary file storage, format detection,
    dataframe loading, and cleanup in one operation.

    :param file: The uploaded file object from FastAPI
    :param sheet_name: Index of the sheet to use for Excel files, defaults to 0
    :return: Dictionary containing processed dataset information including
            filename, row count, columns, and dataset type
    :raises InvalidFileError: If the file has no filename or is invalid
    :raises UnsupportedFormatError: If the file format is not supported
    """
    if not file.filename:
        raise InvalidFileError

    file_extension = os.path.splitext(file.filename)[1].lower().lstrip(".")
    try:
        file_format: Final = FileFormat(file_extension)
        logger.trace("Detected file format: {}", file_format)
    except ValueError as e:
        raise UnsupportedFormatError from e

    temp_dir: Final = tempfile.mkdtemp()
    temp_path: Final = os.path.join(temp_dir, file.filename)

    try:
        content = await file.read()
        Path(temp_path).write_bytes(content)
        await file.seek(0)

        return _process_upload(temp_path, file_format, file.filename, sheet_name)
    finally:
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.trace("Removed temporary directory: {}", temp_dir)
        except Exception as e:
            logger.warning("Failed to clean up temporary files: {}", str(e))


def _process_upload(
    file_path: str,
    file_format: FileFormat,
    original_filename: str,
    sheet_name: int = 0,
) -> dict[str, Any]:
    """Process uploaded file and save as CSV.

    :param file_path: Path to the temporary file
    :param file_format: Format of the uploaded file
    :param original_filename: Original filename from the upload
    :param sheet_name: Sheet index for Excel files, defaults to 0
    :return: Dictionary with dataset information
    :raises EmptyCSVError: If the dataset is empty
    :raises InvalidCSVFormatError: If the file has invalid format
    """
    try:
        df: Final = _load_dataframe(file_path, file_format, sheet_name)

        _validate_dataframe(df)
        dataset_type: Final = _classify_dataset(df)

        csv_filename: Final = _generate_unique_filename(original_filename)
        _save_dataframe(df, csv_filename)

        return {
            "filename": csv_filename,
            "rows": len(df),
            "columns": df.columns.tolist(),
            "dataset_type": dataset_type,
        }
    except Exception:
        raise


def _validate_dataframe(df: pd.DataFrame) -> None:
    """Validate dataframe structure and content.

    :param df: The pandas DataFrame to validate
    :raises EmptyCSVError: If the dataframe is empty
    :raises InvalidCSVFormatError: If the dataframe has less than the minimum
    required columns
    """
    if df.empty:
        raise EmptyCSVError

    if len(df.columns) < MINIMUM_COLUMNS:
        raise InvalidCSVFormatError


def _classify_dataset(df: pd.DataFrame) -> Classification:
    """Classify dataset based on column structure.

    :param df: The pandas DataFrame to classify
    :return: String constant representing the dataset type from DatasetType enum
    """
    columns: Final = {col.lower() for col in df.columns}
    if {"id", "anchor", "positive", "negative"}.issubset(columns):
        return Classification.SENTENCE_TRIPLES

    col_count: Final = len(df.columns)
    if col_count == MINIMUM_COLUMNS:
        return Classification.SENTENCE_DUPLES
    if col_count == MAXIMUM_COLUMNS:
        return Classification.SENTENCE_TRIPLES
    raise InvalidCSVFormatError


def _detect_delimiter(file_path: str) -> Literal[",", ";", "\t", "|"]:
    """Auto-detect CSV delimiter from file.

    :param file_path: Path to the CSV file
    :return: Detected delimiter character or default delimiter if detection fails
    """
    try:
        with open(file_path, newline="", encoding="utf-8") as csvfile:
            sample: Final = csvfile.read(4096)
            if not sample:
                return DEFAULT_DELIMITER

            try:
                dialect = Sniffer().sniff(sample)
                return dialect.delimiter
            except Exception:
                for delimiter in [",", ";", "\t", "|"]:
                    if delimiter in sample:
                        return delimiter
                return DEFAULT_DELIMITER
    except Exception:
        return DEFAULT_DELIMITER


def _load_dataframe(
    file_path: str,
    file_format: FileFormat,
    sheet_name: int = 0,
) -> pd.DataFrame:
    """Load file into pandas DataFrame based on format.

    :param file_path: Path to the file to load
    :param file_format: Format of the file (CSV, JSON, XML, Excel, etc.)
    :param sheet_name: Index of the sheet to use for Excel files, defaults to 0
    :return: Pandas DataFrame containing the loaded data
    :raises EmptyCSVError: If the file contains no data
    :raises InvalidFileError: If the file cannot be parsed correctly
    :raises UnsupportedFormatError: If the file format is not supported
    """
    try:
        match file_format:
            case FileFormat.CSV:
                return _load_csv(file_path)
            case FileFormat.JSON:
                return _load_json(file_path)
            case FileFormat.XML:
                return pd.read_xml(file_path)
            case FileFormat.EXCEL | FileFormat.EXCEL_LEGACY:
                return pd.read_excel(file_path, sheet_name=sheet_name)
            case _:
                raise UnsupportedFormatError("Unsupported file format: {}", file_format)

    except pd.errors.EmptyDataError as e:
        raise EmptyCSVError from e
    except pd.errors.ParserError as e:
        raise InvalidFileError from e


def _load_csv(file_path: str) -> pd.DataFrame:
    """Load CSV file with multiple encoding attempts.

    :param file_path: Path to the CSV file to load
    :return: Pandas DataFrame containing the parsed CSV data
    :raises UnicodeDecodeError: If all encoding attempts fail
    :raises Exception: For other parsing errors
    """
    delimiter: Final = _detect_delimiter(file_path)

    for encoding in ["utf-8", "latin1", "cp1252"]:
        try:
            return pd.read_csv(
                file_path,
                delimiter=delimiter,
                encoding=encoding,
            )
        except UnicodeDecodeError:
            logger.debug("CSV encoding {} failed, trying next encoding", encoding)
            continue
        except Exception as e:
            logger.debug("CSV parsing with {} encoding failed: {}", encoding, str(e))
            continue

    return pd.read_csv(
        file_path,
        delimiter=delimiter,
        encoding="latin1",
        on_bad_lines="skip",
        engine="python",
    )


def _load_json(file_path: str) -> pd.DataFrame:
    """Load JSON file into DataFrame.

    :param file_path: Path to the JSON file
    :return: Pandas DataFrame containing the parsed JSON data
    :raises Exception: If the JSON cannot be parsed in any supported format
    """
    try:
        return pd.read_json(file_path)
    except Exception:
        with open(file_path, encoding="utf-8") as f:
            data: Final = json.load(f)
            if isinstance(data, list):
                return pd.DataFrame(data)
            if isinstance(data, dict) and "data" in data:
                return pd.DataFrame(data["data"])
            return pd.json_normalize(data)


def _generate_unique_filename(original_filename: str) -> str:
    """Generate unique filename with timestamp.

    :param original_filename: The original filename to base the new name on
    :return: A unique filename string with timestamp and .csv extension
    """
    base_name: Final = os.path.splitext(original_filename)[0]
    timestamp: Final = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.csv"


def _save_dataframe(df: pd.DataFrame, filename: str) -> Path:
    """Save DataFrame as CSV in the upload directory.

    :param df: The pandas DataFrame to save
    :param filename: The filename to use for the saved CSV file
    :return: The Path object pointing to the saved file
    """
    file_path: Final = UPLOAD_DIR / filename
    df.to_csv(file_path, index=False)
    logger.trace("Saved dataset to {}", file_path)
    return file_path
