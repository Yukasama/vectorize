"""Dataset router."""

from typing import Annotated, Final

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
from pydantic import BaseModel, Field
from sqlmodel.ext.asyncio.session import AsyncSession

from txt2vec.config.db import get_session

from .service import upload_file

__all__ = ["router"]

router = APIRouter(tags=["Dataset", "Upload"])


class DatasetUploadOptions(BaseModel):
    """Options for dataset upload."""

    question_name: str | None = Field(
        default=None, description="Column name for the question"
    )
    positive_name: str | None = Field(
        default=None, description="Column name for the positive example"
    )
    negative_name: str | None = Field(
        default=None, description="Column name for the negative example"
    )
    sheet_index: int = Field(default=0, description="Sheet index for Excel files")


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
    column_mapping = {
        "question": options.question_name,
        "positive": options.positive_name,
        "negative": options.negative_name,
    }

    logger.debug("Dataset upload started", file=file.filename)
    dataset_id: Final = await upload_file(db, file, column_mapping, options.sheet_index)
    logger.debug("Dataset uploaded", datasetId=dataset_id)

    return Response(
        status_code=status.HTTP_201_CREATED,
        headers={"Location": f"{request.url}/{dataset_id}"},
    )
