"""Dataset router."""

from typing import Annotated, Final

from fastapi import APIRouter, File, Request, Response, UploadFile, status
from loguru import logger

from txt2vec.datasets.service import upload_file

__all__ = ["router"]

router = APIRouter(tags=["Dataset", "Upload"])


@router.post("")
async def upload_dataset(
    file: Annotated[UploadFile, File()],
    request: Request,
    sheet_name: int = 0,
) -> Response:
    """Upload a dataset file and convert it to CSV format.

    :param file: The file to upload (CSV, JSON, XML, or Excel)
    :param request: The HTTP request object
    :param sheet_name: Sheet index for Excel files, by default 0

    :return: OK response with the dataset ID in the Location header
    """
    logger.debug("Dataset upload started", file=file.filename)
    dataset_id: Final = await upload_file(file, sheet_name)
    logger.debug("Dataset uploaded", datasetId=dataset_id)

    return Response(
        status_code=status.HTTP_201_CREATED,
        headers={"Location": f"{request.url}/{dataset_id}"},
    )
