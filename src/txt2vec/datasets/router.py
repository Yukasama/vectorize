"""Dataset router."""

from typing import Annotated, Final
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    Request,
    Response,
    UploadFile,
    status,
)
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from txt2vec.config.db import get_session

from .models import Dataset
from .repository import get_all_datasets, get_dataset
from .service import upload_file
from .upload_options_model import DatasetUploadOptions

__all__ = ["router"]


router = APIRouter(tags=["Dataset", "Upload"])


@router.get("")
async def get_datasets(
    db: Annotated[AsyncSession, Depends(get_session)],
) -> list[dict]:
    """Retrieve all datasets with limited fields.

    Args:
        db: Database session for persistence operations

    Returns:
        List of datasets with limited fields (name, file_name, classification,
        created_at)
    """
    datasets = await get_all_datasets(db)

    return [
        {
            "id": str(dataset.id),
            "name": dataset.name,
            "file_name": dataset.file_name,
            "classification": dataset.classification,
            "created_at": dataset.created_at,
        }
        for dataset in datasets
    ]


@router.get("/{dataset_id}")
async def get_dataset_by_id(
    dataset_id: UUID,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> Dataset:
    """Retrieve a single dataset by its ID.

    Args:
        dataset_id: The UUID of the dataset to retrieve
        db: Database session for persistence operations

    Returns:
        The dataset object

    Raises:
        DatasetNotFoundError: If the dataset with the specified ID doesn't exist
    """
    dataset = await get_dataset(db, dataset_id)

    return {
        "id": str(dataset.id),
        "name": dataset.name,
        "file_name": dataset.file_name,
        "classification": dataset.classification,
        "created_at": dataset.created_at,
        "updated_at": dataset.updated_at,
        "rows": dataset.rows,
        "synthesis_id": dataset.synthesis_id,
    }


@router.post("")
async def upload_dataset(
    file: Annotated[UploadFile, File()],
    request: Request,
    db: Annotated[AsyncSession, Depends(get_session)],
    options: Annotated[DatasetUploadOptions, Depends()],
) -> Response:
    """Upload a dataset file and convert it to CSV format.

    Args:
        file: The file to upload (CSV, JSON, XML, or Excel)
        request: The HTTP request object
        db: Database session for persistence operations
        options: Options for dataset upload, including column names and sheet index

    Returns:
        Response with status code 201 and the dataset ID in the Location header
    """
    dataset_id: Final = await upload_file(db, file, options)
    logger.debug("Dataset uploaded", datasetId=dataset_id)

    return Response(
        status_code=status.HTTP_201_CREATED,
        headers={"Location": f"{request.url}/{dataset_id}"},
    )
