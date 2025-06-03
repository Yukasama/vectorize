"""Router for synthetic data generation from media files."""

from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.config.db import get_session

from .models import SynthesisTask
from .repository import (
    get_synthesis_task_by_id,
    get_synthesis_tasks,
    save_synthesis_task,
)
from .service import (
    validate_existing_dataset,
    validate_upload_request,
)
from .tasks import (
    process_existing_dataset_background,
    process_file_contents_background,
)

__all__ = ["router"]

router = APIRouter(tags=["Synthesis"])


@router.post("/media", status_code=status.HTTP_202_ACCEPTED)
async def upload_media_for_synthesis(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Annotated[AsyncSession, Depends(get_session)],
    files: Annotated[
        list[UploadFile] | None,
        File(description="Image or PDF files for text extraction"),
    ] = None,
    existing_dataset_id: Annotated[
        str | None, Form(description="Optional existing dataset ID to use")
    ] = None,
) -> dict[str, str | UUID | int]:
    """Upload media files to extract text and create synthetic datasets.

    The processing is done in the background.

    Args:
        request: HTTP request object
        background_tasks: Background task manager
        db: Database session
        files: List of image or PDF files to process
        existing_dataset_id: Optional ID of existing dataset to use

    Returns:
        Dictionary with task information and status URL
    """
    validate_upload_request(files, existing_dataset_id)

    task = SynthesisTask()
    task = await save_synthesis_task(db, task)

    if existing_dataset_id:
        dataset_uuid = await validate_existing_dataset(db, existing_dataset_id)

        background_tasks.add_task(
            process_existing_dataset_background,
            task_id=task.id,
            dataset_id=dataset_uuid,
        )

        logger.info(
            "Synthesis task created using existing dataset.",
            taskId=task.id,
            datasetId=dataset_uuid,
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
            "dataset_id": dataset_uuid,
        }

    file_contents = []
    if files:
        for file in files:
            if not file.filename:
                continue
            try:
                content = await file.read()
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

    background_tasks.add_task(
        process_file_contents_background,
        task_id=task.id,
        file_contents=file_contents,
        options=None,
    )

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
