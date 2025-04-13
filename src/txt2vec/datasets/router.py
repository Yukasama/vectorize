"""Dataset router handling file uploads."""

from typing import Annotated, Final

from fastapi import APIRouter, Depends, File, Request, Response, UploadFile, status
from loguru import logger

from txt2vec.datasets.schemas import DatasetUploadResponse
from txt2vec.datasets.service import DatasetService
from txt2vec.handle_exceptions import handle_exceptions

__all__ = ["router"]

router = APIRouter(prefix="/datasets", tags=["Dataset"])


@router.post(
    "/", response_model=DatasetUploadResponse, status_code=status.HTTP_201_CREATED
)
@handle_exceptions
async def upload_dataset(
    file: Annotated[UploadFile, File()],
    service: Annotated[DatasetService, Depends()],
    request: Request,
    sheet_name: int = 0,
) -> Response:
    """Upload a dataset file and convert it to CSV format.

    :param file: The file to upload (CSV, JSON, XML, or Excel)
    :param service: Service for dataset operations
    :param request: The HTTP request object
    :param sheet_name: Sheet index for Excel files, by default 0

    :return: Dataset information including filename, size, preview and classification
    """
    logger.debug("file={}", file.filename)
    result: Final = await service.upload_file(file, sheet_name)
    logger.debug("(done): file={}", result["filename"])

    return Response(
        status_code=status.HTTP_201_CREATED,
        headers={"Location": f"{request.url}/{1}"},
    )
