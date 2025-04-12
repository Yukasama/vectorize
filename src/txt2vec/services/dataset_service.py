import os
from csv import Sniffer
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class DatasetResponse(BaseModel):
    """Response model for dataset operations"""

    filename: str
    rows: int
    columns: list[str]
    preview: list[dict[str, Any]]


class FileFormat(str, Enum):
    """Supported file formats"""

    CSV = "csv"
    JSON = "json"
    XML = "xml"
    EXCEL = "xlsx"
    EXCEL_LEGACY = "xls"


class DatasetService:
    """Service class for dataset operations following the service pattern"""

    def __init__(self, upload_dir: Path = UPLOAD_DIR):
        """Initialize dataset service with configurable upload directory"""
        self.upload_dir = upload_dir
        # Ensure the upload directory exists
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def detect_delimiter(file_path: str) -> str:
        """Auto-detect CSV delimiter using csv.Sniffer"""
        try:
            with open(file_path, newline='') as csvfile:
                sample = csvfile.read(4096)
                sniffer = Sniffer()
                if not sample:
                    return ','  # Default delimiter if file is empty
                return sniffer.sniff(sample).delimiter
        except Exception:
            return ','  # Fall back to comma if detection fails

    @staticmethod
    def load_dataframe(
        file_path: str,
        file_format: FileFormat,
        sheet_name: int | None = 0,
    ) -> pd.DataFrame:
        """Load data into DataFrame based on file format"""
        if file_format == FileFormat.CSV:
            delimiter = DatasetService.detect_delimiter(file_path)
            return pd.read_csv(file_path, delimiter=delimiter)
        if file_format == FileFormat.JSON:
            return pd.read_json(file_path)
        if file_format == FileFormat.XML:
            return pd.read_xml(file_path)
        if file_format in [FileFormat.EXCEL, FileFormat.EXCEL_LEGACY]:
            return pd.read_excel(file_path, sheet_name=sheet_name)
        raise ValueError(f"Unsupported file format: {file_format}")

    @staticmethod
    def generate_unique_filename(original_filename: str) -> str:
        """Generate unique filename with timestamp"""
        base_name = os.path.splitext(original_filename)[0]
        return f"{base_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"

    @staticmethod
    def save_dataframe(df: pd.DataFrame, filename: str) -> Path:
        """Save DataFrame as CSV"""
        file_path = UPLOAD_DIR / filename
        df.to_csv(file_path, index=False)
        return file_path

    @staticmethod
    def create_response(df: pd.DataFrame, filename: str) -> DatasetResponse:
        """Create standardized response object"""
        return DatasetResponse(
            filename=filename,
            rows=len(df),
            columns=df.columns.tolist(),
            preview=df.head(5).to_dict(orient="records"),
        )
