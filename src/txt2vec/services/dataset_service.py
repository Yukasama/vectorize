import os
from csv import Sniffer
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from txt2vec.services.exceptions import (
    InvalidCSVFormatException,
    InvalidFileException,
    ProcessingErrorException,
    UnsupportedFormatException,
)
from txt2vec.services.models import DatasetType, FileFormat


class DatasetService:
    """Service class for dataset operations."""

    def __init__(self, upload_dir: Path | None = None):
        """Initialize dataset service with configurable upload directory."""
        self.upload_dir = upload_dir or Path("data/uploads")
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def process_upload(
        self,
        file_path: str,
        file_format: FileFormat,
        original_filename: str,
        sheet_name: int | None = 0,
    ) -> dict[str, Any]:
        """Process uploaded file and save as CSV."""
        try:
            df = self.load_dataframe(file_path, file_format, sheet_name)
            self.validate_dataframe(df)
            csv_filename = self.generate_unique_filename(original_filename)
            dataset_type = self.classify_dataset(df)

            return {
                "filename": csv_filename,
                "rows": len(df),
                "columns": df.columns.tolist(),
                "dataset_type": dataset_type,
            }
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            if isinstance(
                e,
                (
                    InvalidFileException,
                    InvalidCSVFormatException,
                    UnsupportedFormatException,
                ),
            ):
                raise e
            raise ProcessingErrorException(f"Error processing file: {e!s}")

    def validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate that the dataframe has the expected columns and format"""
        if df.empty or len(df.columns) < 2 or len(df.columns) > 4:
            raise InvalidCSVFormatException

    def classify_dataset(self, df: pd.DataFrame) -> str:
        """Classify the dataset based on its structure

        Rules:
        - If exactly 2 columns: SENTENCE_DUPLES
        - If exactly 3 columns: SENTENCE_TRIPLES
        - If columns match [id, anchor, positive, negative]: TRIPLET_DATASET
        - Otherwise: UNKNOWN
        """
        columns = set(df.columns.str.lower())

        # Check for specific triplet dataset pattern
        if set(["id", "anchor", "positive", "negative"]).issubset(columns):
            return DatasetType.TRIPLET_DATASET

        # Check based on column count
        col_count = len(df.columns)
        if col_count == 2:
            return DatasetType.SENTENCE_DUPLES
        elif col_count == 3:
            return DatasetType.SENTENCE_TRIPLES
        else:
            return DatasetType.UNKNOWN

    @staticmethod
    def detect_delimiter(file_path: str) -> str:
        """Auto-detect CSV delimiter using csv.Sniffer"""
        try:
            with open(file_path, newline="", encoding="utf-8") as csvfile:
                sample = csvfile.read(4096)
                if not sample:
                    return ","

                sniffer = Sniffer()
                try:
                    dialect = sniffer.sniff(sample)
                    return dialect.delimiter
                except:
                    for delimiter in [",", ";", "\t", "|"]:
                        if delimiter in sample:
                            return delimiter
                    return ","
        except Exception as e:
            logger.warning(f"Delimiter detection failed: {e}. Using default ','")
            return ","

    def load_dataframe(
        self,
        file_path: str,
        file_format: FileFormat,
        sheet_name: int | None = 0,
    ) -> pd.DataFrame:
        """Load data into DataFrame based on file format"""
        try:
            if file_format == FileFormat.CSV:
                delimiter = self.detect_delimiter(file_path)
                logger.info(f"Detected delimiter: '{delimiter}'")

                encodings = ["utf-8", "latin1", "cp1252"]
                for encoding in encodings:
                    try:
                        return pd.read_csv(
                            file_path,
                            delimiter=delimiter,
                            encoding=encoding,
                            on_bad_lines="warn",
                        )
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        logger.warning(f"Error with encoding {encoding}: {e}")
                        continue

                return pd.read_csv(
                    file_path,
                    delimiter=delimiter,
                    encoding="latin1",
                    on_bad_lines="skip",
                    engine="python",
                )

            elif file_format == FileFormat.JSON:
                try:
                    return pd.read_json(file_path)
                except:
                    with open(file_path, encoding="utf-8") as f:
                        import json

                        data = json.load(f)
                        if isinstance(data, list):
                            return pd.DataFrame(data)
                        elif isinstance(data, dict) and "data" in data:
                            return pd.DataFrame(data["data"])
                        else:
                            return pd.json_normalize(data)

            elif file_format == FileFormat.XML:
                return pd.read_xml(file_path)

            elif file_format in [FileFormat.EXCEL, FileFormat.EXCEL_LEGACY]:
                return pd.read_excel(file_path, sheet_name=sheet_name)

            raise UnsupportedFormatException(f"Unsupported file format: {file_format}")

        except pd.errors.EmptyDataError:
            raise InvalidFileException("The file is empty")
        except pd.errors.ParserError:
            raise InvalidFileException("The file could not be parsed correctly")
        except Exception as e:
            logger.error(f"Error loading dataframe: {e}")
            raise InvalidFileException(f"Error loading file: {e!s}")

    @staticmethod
    def generate_unique_filename(original_filename: str) -> str:
        """Generate unique filename with timestamp"""
        base_name = os.path.splitext(original_filename)[0]
        return f"{base_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"

    def save_dataframe(self, df: pd.DataFrame, filename: str) -> Path:
        """Save DataFrame as CSV"""
        file_path = self.upload_dir / filename
        df.to_csv(file_path, index=False)
        logger.info(f"Saved dataset to {file_path}")
        return file_path
