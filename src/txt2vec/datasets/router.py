"""Dataset router."""

from typing import Annotated
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

from .exceptions import InvalidFileError
from .models import DatasetAll, DatasetPublic, DatasetUpdate
from .service import read_all_datasets, read_dataset, update_dataset_srv, upload_file
from .upload_options_model import DatasetUploadOptions
from .utils.validate_zip import handle_zip_upload

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


@router.put("/{dataset_id}")
async def put_dataset(
    dataset_id: UUID,
    request: Request,
    dataset: DatasetUpdate,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    """Update a dataset with version control using ETags.

    Updates a dataset by its ID, requiring an If-Match header with the current
    version to prevent concurrent modification issues.

    Args:
        dataset_id: The UUID of the dataset to update
        request: The HTTP request object containing If-Match header
        response: FastAPI response object for setting headers
        dataset: The updated dataset object
        db: Database session for persistence operations

    Returns:
        204 No Content response with Location header

    Raises:
        VersionMismatchError: If the ETag doesn't match current version
        VersionMissingError: If the If-Match header is missing
        DatasetNotFoundError: If the dataset doesn't exist
    """
    new_version = await update_dataset_srv(db, request, dataset_id, dataset)
    logger.debug("Dataset updated", datasetId=dataset_id)

    return Response(
        status_code=status.HTTP_204_NO_CONTENT,
        headers={"Location": f"{request.url.path}", "ETag": f'"{new_version}"'},
    )


@router.post("")
async def upload_dataset(
    files: Annotated[list[UploadFile], File(description="One or many files")],
    request: Request,
    db: Annotated[AsyncSession, Depends(get_session)],
    options: Annotated[DatasetUploadOptions, Depends()],
) -> Response:
    """Upload a dataset file and convert it to CSV format.

    Args:
        files: The files to upload (CSV, JSON, XML, Excel, or ZIP)
        request: The HTTP request object
        db: Database session for persistence operations
        options: Options for dataset upload, including column names and sheet index

    Returns:
        Response with status code 201 and the dataset ID in the Location header
    """
    if not files:
        raise InvalidFileError

    first = files[0]

    if len(files) == 1 and first.filename.lower().endswith(".zip"):
        dataset_ids = await handle_zip_upload(first, db, options)

    else:
        if any(f.filename.lower().endswith(".zip") for f in files):
            raise InvalidFileError("Cannot mix ZIP and individual files")

        dataset_ids = [await upload_file(db, file, options) for file in files]

    last_id = dataset_ids[-1]
    return Response(
        headers={"Location": f"{request.url}/{last_id}"},
        status_code=status.HTTP_201_CREATED,
    )
