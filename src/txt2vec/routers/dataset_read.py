import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from loguru import logger
from pydantic import BaseModel

from txt2vec.services.dataset_service import DatasetService, FileFormat

UPLOAD_DIR = Path("data/uploads")


class ErrorCode(str, Enum):
    """Error codes for standardized error handling"""

    INVALID_FILE = "INVALID_FILE"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    PROCESSING_ERROR = "PROCESSING_ERROR"
    NOT_FOUND = "NOT_FOUND"
    SERVER_ERROR = "SERVER_ERROR"


class DatasetResponse(BaseModel):
    """Response model for dataset operations"""

    filename: str
    rows: int
    columns: list[str]
    preview: list[dict[str, Any]]


# Dependency for error handling
def get_dataset_service():
    """Dependency injection for DatasetService"""
    return DatasetService()


# Router setup
router = APIRouter(prefix="/dataset", tags=["dataset"])


# Exception handler using decorator pattern
def handle_dataset_exceptions(func):
    """Decorator for standardized exception handling"""

    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            # Re-raise FastAPI HTTP exceptions
            raise
        except FileNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": ErrorCode.NOT_FOUND, "message": "Dataset not found"},
            )
        except ValueError as e:
            if "Unsupported file format" in str(e):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"code": ErrorCode.UNSUPPORTED_FORMAT, "message": str(e)},
                )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": ErrorCode.INVALID_FILE, "message": str(e)},
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "code": ErrorCode.SERVER_ERROR,
                    "message": f"An unexpected error occurred: {e!s}",
                },
            )

    return wrapper


@router.post("/", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
@handle_dataset_exceptions
async def upload_dataset(
    file: UploadFile = File(...),
    sheet_name: int | None = 0,
    service: DatasetService = Depends(get_dataset_service),
):
    """Upload a dataset file and convert it to CSV format.

    The delimiter for CSV files is automatically detected.

    Parameters
    ----------
    - file: The file to upload (CSV, JSON, XML, or Excel)
    - sheet_name: Sheet index for Excel files (default: 0)

    Returns
    -------
    - Information about the processed dataset

    """
    logger.debug("file: {}", file)
    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": ErrorCode.INVALID_FILE, "message": "No filename provided"},
        )

    # Get file extension and validate format
    file_extension = file.filename.split(".")[-1].lower()
    try:
        file_format = FileFormat(file_extension)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": ErrorCode.UNSUPPORTED_FORMAT,
                "message": f"Unsupported file format: {file_extension}. Supported formats: {', '.join([f.value for f in FileFormat])}",
            },
        )

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
        content = await file.read()
        temp_file.write(content)

    try:
        # Process the file using service layer
        df = service.load_dataframe(temp_path, file_format, sheet_name)
        csv_filename = service.generate_unique_filename(file.filename)
        service.save_dataframe(df, csv_filename)
        return service.create_response(df, csv_filename)
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@router.get("/", response_model=list[str])
@handle_dataset_exceptions
async def list_datasets():
    """List all available datasets"""
    files = [f.name for f in UPLOAD_DIR.glob("*.csv")]
    return files


@router.get("/{filename}", response_model=DatasetResponse)
@handle_dataset_exceptions
async def get_dataset(
    filename: str, service: DatasetService = Depends(get_dataset_service),
):
    """Get information about a specific dataset"""
    file_path = UPLOAD_DIR / filename

    if not file_path.exists() or not filename.endswith(".csv"):
        raise FileNotFoundError(f"Dataset {filename} not found")

    df = pd.read_csv(file_path)
    return service.create_response(df, filename)


@router.delete("/{filename}", status_code=status.HTTP_204_NO_CONTENT)
@handle_dataset_exceptions
async def delete_dataset(filename: str):
    """Delete a dataset"""
    file_path = UPLOAD_DIR / filename

    if not file_path.exists() or not filename.endswith(".csv"):
        raise FileNotFoundError(f"Dataset {filename} not found")

    os.remove(file_path)
