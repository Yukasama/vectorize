"""Dataset router."""

from typing import Annotated, Final

from fastapi import APIRouter, File, Request, Response, UploadFile, status
from loguru import logger

from txt2vec.datasets.service import upload_file
from txt2vec.handle_exceptions import handle_exceptions

__all__ = ["router"]

router = APIRouter(tags=["Dataset"])


@router.post("")
@handle_exceptions
async def upload_dataset(
    file: Annotated[UploadFile, File()],
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
    result: Final = await upload_file(file, sheet_name)
    logger.debug("(done): file={}", result["filename"])

    return Response(
        status_code=status.HTTP_201_CREATED,
        headers={"Location": f"{request.url}/{1}"},
    )
