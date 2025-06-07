"""Process dataset uploads."""

from uuid import UUID

from fastapi import (
    UploadFile,
)
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.common.app_error import AppError

from ..schemas import DatasetUploadOptions
from ..service import upload_dataset_svc

__all__ = ["_process_uploads"]


async def _process_uploads(
    files: list[UploadFile],
    db: AsyncSession,
    options: DatasetUploadOptions,
) -> tuple[list[UUID], list[dict]]:
    """Upload a list of files and collect successes and failures.

    Args:
        files: Files already validated for mixing rules (e.g., no ZIP mixed
            with regular files).
        db: Active async database session used by ``upload_dataset_svc``.
        options: Client-supplied upload options forwarded to
            ``upload_dataset_svc``.

    Returns:
        A tuple with two items:
            * List of dataset IDs for successfully processed files.
            * List of dictionaries describing each failed upload,
              e.g. ``{"filename": "bad.csv", "error": "Invalid header"}``.

    Raises:
        AppError: Re-raised when only one file was supplied and that upload
            failed.
    """
    dataset_ids: list[UUID] = []
    failed_uploads: list[dict] = []

    for file in files:
        try:
            dataset_ids.append(await upload_dataset_svc(db, file, options))
        except AppError as e:
            if len(files) == 1:
                raise e
            failed_uploads.append({"filename": file.filename, "error": str(e.message)})
        except Exception:
            failed_uploads.append({
                "filename": file.filename,
                "error": "An unknown error occurred",
            })

    return dataset_ids, failed_uploads
