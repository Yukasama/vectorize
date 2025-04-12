import os
import shutil
import tempfile
from typing import Any

from fastapi import APIRouter, Depends, File, UploadFile, status
from loguru import logger
from pydantic import BaseModel

from txt2vec.handle_exceptions import handle_exceptions
from txt2vec.services.dataset_service import DatasetService, FileFormat
from txt2vec.services.exceptions import (
    InvalidFileException,
    UnsupportedFormatException,
)

router = APIRouter(prefix="/dataset", tags=["dataset"])


class DatasetResponse(BaseModel):
    """Response model for dataset operations."""

    filename: str
    rows: int
    columns: list[str]
    dataset_type: str


def get_dataset_service() -> DatasetService:
    """Dependency injection for DatasetService."""
    return DatasetService()


@router.post("/", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
@handle_exceptions
async def upload_dataset(
    file: UploadFile = File(...),
    sheet_name: int = 0,
    service: DatasetService = Depends(get_dataset_service),
) -> dict[str, Any]:
    """Upload a dataset file and convert it to CSV format.

    The system will automatically:
    - Detect the file format (.csv, .json, .xml, .xlsx, .xls)
    - Detect CSV delimiters if applicable
    - Classify the dataset type based on its structure
    - Save the dataset as CSV for further processing

    Parameters
    ----------
    - file: The file to upload (CSV, JSON, XML, or Excel)
    - sheet_name: Sheet index for Excel files (default: 0)

    Returns
    -------
    - Dataset information including filename, size, preview and classification

    """
    logger.info(f"Processing upload: {file.filename}")

    if not file.filename:
        raise InvalidFileException

    file_extension = os.path.splitext(file.filename)[1].lower().lstrip(".")
    try:
        file_format = FileFormat(file_extension)
        logger.info(f"Detected file format: {file_format}")
    except ValueError:
        raise UnsupportedFormatException

    # Use a more reliable way to save uploaded file
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)

    try:
        # Save uploaded file to temp location
        with open(temp_path, "wb") as buffer:
            # Read in chunks to avoid memory issues with large files
            chunk_size = 1024 * 1024  # 1MB chunks
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                buffer.write(chunk)

        # Reset file position for potential reuse
        await file.seek(0)

        logger.info(f"Saved upload to temporary file: {temp_path}")

        # Process the file using service layer
        result = service.process_upload(
            temp_path,
            file_format,
            file.filename,
            sheet_name,
        )

        logger.info(f"File processed successfully: {result['filename']}")
        return result

    except Exception as e:
        logger.error(f"Error during file upload: {e!s}")
        # Re-raise to let the exception handler catch it
        raise

    finally:
        # Clean up temporary directory and its contents
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.debug(f"Removed temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary files: {e!s}")
