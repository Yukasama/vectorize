"""Router for synthetic data generation from media files."""

from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.config.db import get_session
from vectorize.dataset.repository import get_dataset_db

from .models import SynthesisTask
from .repository import (
    get_synthesis_task_by_id,
    get_synthesis_tasks,
    save_synthesis_task,
)
from .tasks import (
    process_existing_dataset_background_bg,
    process_file_contents_background_bg,
)

__all__ = ["router"]


router = APIRouter(tags=["Synthesis"])


@router.post("/media", status_code=status.HTTP_202_ACCEPTED)
async def upload_media_for_synthesis(  # noqa: PLR0912
    request: Request,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> dict[str, str | UUID | int]:
    """Upload media files to extract text and create synthetic datasets.

    The processing is done in the background.

    Args:
        request: HTTP request object
        db: Database session

    Returns:
        Dictionary with task information and status URL
    """
    form = await request.form()

    dataset_id = None
    if "dataset_id" in form:
        dataset_id_value = form["dataset_id"]
        if isinstance(dataset_id_value, str):
            try:
                dataset_id = UUID(dataset_id_value)
            except (ValueError, TypeError) as e:
                raise HTTPException(
                    status_code=422,
                    detail="Invalid dataset_id format. Must be a valid UUID.",
                ) from e
        else:
            raise HTTPException(
                status_code=422,
                detail="dataset_id must be a string value, not a file.",
            )

    files = []
    for key, value in form.multi_items():
        if (
            key == "files"
            and isinstance(value, UploadFile)
            and value.filename
            and value.size
            and value.size > 0
        ):
            files.append(value)

    if not files and dataset_id is None:
        raise HTTPException(
            status_code=422,
            detail="Either files or existing dataset id must be provided.",
        )

    task = await save_synthesis_task(db, SynthesisTask())

    if dataset_id and not files:
        dataset_db = await get_dataset_db(db, dataset_id)
        process_existing_dataset_background_bg.send(str(task.id), str(dataset_db.id))

        logger.info(
            "Synthesis task created using existing dataset.",
            taskId=task.id,
            datasetId=dataset_db.id,
        )

        return {
            "message": (
                "Synthetic task with existing dataset created, "
                "processing in background."
            ),
            "task_id": task.id,
            "status_url": str(
                request.url_for("get_synthesis_task_info", task_id=task.id)
            ),
            "dataset_id": dataset_db.id,
        }

    file_contents = []
    if files:
        for file in files:
            try:
                content = await file.read()
                if content:
                    file_contents.append((file.filename, content))
            except Exception as e:
                logger.error(f"Error reading file {file.filename}: {e}")
            finally:
                await file.close()

    if not file_contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid files provided or all files were empty.",
        )

    process_file_contents_background_bg.send(str(task.id), file_contents)

    logger.info(
        "Synthesis task created, starting background processing.",
        taskId=task.id,
        fileCount=len(file_contents),
    )

    return {
        "message": "Media files upload accepted, processing in background.",
        "task_id": task.id,
        "status_url": str(request.url_for("get_synthesis_task_info", task_id=task.id)),
        "file_count": len(file_contents),
    }


@router.get("/tasks/{task_id}", name="get_synthesis_task_info")
async def get_synthesis_task_info(
    task_id: UUID, db: Annotated[AsyncSession, Depends(get_session)]
) -> SynthesisTask:
    """Retrieves the status and information of a synthesis task.

    Args:
        task_id: UUID of the synthesis task to retrieve
        db: Database session

    Returns:
        SynthesisTask object containing task status and related information

    Raises:
        HTTPException: If task is not found
    """
    task = await get_synthesis_task_by_id(db, task_id)

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Synthesis task not found"
        )

    return task


@router.get("")
async def list_synthesis_tasks(
    db: Annotated[AsyncSession, Depends(get_session)],
    limit: Annotated[
        int, Query(description="Maximum number of tasks to return", ge=1, le=100)
    ] = 20,
) -> list[SynthesisTask]:
    """Retrieves a list of synthesis tasks.

    Args:
        limit: Maximum number of tasks to return (default: 20, max: 100)
        db: Database session

    Returns:
        List of SynthesisTask objects
    """
    return await get_synthesis_tasks(db, limit)
