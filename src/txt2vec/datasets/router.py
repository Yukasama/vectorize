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

from .models import DatasetAll, DatasetPublic
from .service import read_all_datasets, read_dataset, upload_file
from .upload_options_model import DatasetUploadOptions

__all__ = ["router"]


router = APIRouter(tags=["Dataset", "Upload"])


@router.get("")
async def get_datasets(
    db: Annotated[AsyncSession, Depends(get_session)],
) -> list[DatasetAll]:
    """Retrieve all datasets with limited fields.

    Args:
        db: Database session for persistence operations

    Returns:
        List of datasets with limited fields (DatasetAll model)
    """
    datasets = await read_all_datasets(db)
    logger.debug("Datasets retrieved", length=len(datasets))
    return datasets


@router.get("/{dataset_id}")
async def get_dataset_by_id(
    dataset_id: UUID,
    request: Request,
    response: Response,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> DatasetPublic | None:
    """Retrieve a single dataset by its ID.

    Args:
        dataset_id: The UUID of the dataset to retrieve
        request: The HTTP request object
        response: FastAPI response object for setting headers
        db: Database session for persistence operations

    Returns:
        The dataset object

    Raises:
        DatasetNotFoundError: If the dataset with the specified ID doesn't exist
    """
    dataset, version = await read_dataset(db, dataset_id)
    response.headers["ETAG"] = f'"{version}"'

    e_tag = request.headers.get("If-None-Match")
    if e_tag:
        clean_etag = e_tag.strip().strip('"')
        if clean_etag == str(version):
            logger.debug(
                "Dataset not modified",
                datasetId=dataset_id,
                etag=clean_etag,
                version=version,
            )
            response.status_code = status.HTTP_304_NOT_MODIFIED
            return None

    logger.debug("Dataset retrieved", datasetId=dataset_id, version=version)

    return dataset


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
