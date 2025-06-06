"""Background tasks for synthesis processing."""

import tempfile
from pathlib import Path
from uuid import UUID, uuid4

from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.common.task_status import TaskStatus
from vectorize.config import settings
from vectorize.config.db import engine
from vectorize.dataset.classification import Classification
from vectorize.dataset.dataset_source import DatasetSource
from vectorize.dataset.exceptions import DatasetNotFoundError
from vectorize.dataset.models import Dataset
from vectorize.dataset.repository import get_dataset_db, upload_dataset_db
from vectorize.dataset.schemas import DatasetUploadOptions
from vectorize.dataset.utils.dataset_fs import _save_dataframe_to_fs

from .repository import update_synthesis_task_status
from .text_extractor import extract_text_from_media

__all__ = [
    "process_existing_dataset_background",
    "process_file_contents_background",
]


async def process_file_contents_background(
    task_id: UUID,
    file_contents: list[tuple[str, bytes]],
    options: DatasetUploadOptions | None = None,
) -> None:
    """Process file contents extracted from uploaded files.

    Args:
        task_id: ID of the synthesis task
        file_contents: List of tuples containing (filename, file_content)
        options: Optional dataset upload options
    """
    async with AsyncSession(engine, expire_on_commit=False) as db:
        try:
            logger.info(
                "[BG Synthetic] Starting processing of file contents",
                taskId=task_id,
                fileCount=len(file_contents),
            )

            dataset_ids = []

            for filename, content in file_contents:
                try:
                    dataset_id = await _process_single_file(
                        db, task_id, filename, content, options
                    )
                    if dataset_id:
                        dataset_ids.append(dataset_id)

                except Exception as e:
                    logger.error(
                        f"[BG Synthetic] Error processing file {filename}: {e}"
                    )
                    continue

            await _finalize_task_status(db, task_id, dataset_ids)

        except Exception as e:
            logger.error(f"[BG Synthetic] Error in task {task_id}: {e}")
            await update_synthesis_task_status(
                db, task_id, TaskStatus.FAILED, error_msg=str(e)
            )
        finally:
            logger.debug("[BG Synthetic] Database session closed", taskId=task_id)


async def process_existing_dataset_background(
    task_id: UUID,
    dataset_id: UUID,
    options: DatasetUploadOptions | None = None,
) -> None:
    """Process an existing dataset through text extractor to create new synthetic data.

    Args:
        task_id: The synthesis task ID
        dataset_id: ID of existing dataset to use as input
        options: Optional dataset upload options
    """
    async with AsyncSession(engine, expire_on_commit=False) as db:
        try:
            logger.info(
                "[BG Synthetic] Processing existing dataset through text extractor",
                taskId=task_id,
                sourceDatasetId=dataset_id,
            )

            source_dataset = await get_dataset_db(db, dataset_id)
            if not source_dataset:
                raise DatasetNotFoundError(
                    dataset_id=dataset_id, message=f"Dataset {dataset_id} not found"
                )

            dataset_file_path = settings.dataset_upload_dir / source_dataset.file_name

            if not dataset_file_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {dataset_file_path}")

            df = extract_text_from_media(dataset_file_path, "dataset", options)

            classification = (
                Classification.SENTENCE_TRIPLES
                if "rejected" in df.columns
                else Classification.SENTENCE_DUPLES
            )

            unique_name = f"{source_dataset.name}_{uuid4()}.csv"

            _save_dataframe_to_fs(df, unique_name)

            new_dataset = Dataset(
                name=f"{source_dataset.name}",
                file_name=unique_name,
                classification=classification,
                rows=len(df),
                source=DatasetSource.SYNTHETIC,
                synthesis_id=task_id,
            )

            new_dataset_id = await upload_dataset_db(db, new_dataset)

            await update_synthesis_task_status(db, task_id, TaskStatus.DONE)

            logger.info(
                "[BG Synthetic] Synthetic dataset created successfully",
                taskId=task_id,
                sourceDatasetId=dataset_id,
                newDatasetId=new_dataset_id,
                syntheticRows=len(df),
            )

        except Exception as e:
            logger.error(f"[BG Synthetic] Error processing dataset {dataset_id}: {e}")
            await update_synthesis_task_status(
                db, task_id, TaskStatus.FAILED, error_msg=str(e)
            )
        finally:
            logger.debug("[BG Synthetic] Database session closed", taskId=task_id)


async def _process_single_file(
    db: AsyncSession,
    task_id: UUID,
    filename: str,
    content: bytes,
    options: DatasetUploadOptions | None,
) -> UUID | None:
    """Process a single file and create dataset.

    Args:
        db: Database session
        task_id: Synthesis task ID
        filename: Name of the file
        content: File content bytes
        options: Upload options

    Returns:
        Dataset ID if successful, None otherwise
    """
    file_path = Path(filename)
    ext = file_path.suffix.lower().lstrip(".")

    if ext not in {"png", "jpg", "jpeg", "pdf"}:
        logger.warning(
            f"[BG Synthetic] Unsupported file format: {ext}",
            filename=filename,
        )
        return None

    file_size = len(content)
    if file_size > settings.dataset_max_upload_size:
        logger.warning(
            f"[BG Synthetic] File too large: {file_size} bytes",
            filename=filename,
            maxSize=settings.dataset_max_upload_size,
        )
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.write(content)

    try:
        df = extract_text_from_media(temp_path, ext, options)

        classification = (
            Classification.SENTENCE_TRIPLES
            if "rejected" in df.columns
            else Classification.SENTENCE_DUPLES
        )

        unique_name = f"{file_path.stem}_{uuid4()}.csv"

        _save_dataframe_to_fs(df, unique_name)

        dataset = Dataset(
            name=file_path.stem,
            file_name=unique_name,
            classification=classification,
            source=DatasetSource.SYNTHETIC,
            rows=len(df),
            synthesis_id=task_id,
        )

        dataset_id = await upload_dataset_db(db, dataset)

        logger.debug(
            "[BG Synthetic] Processed file successfully",
            filename=filename,
            datasetId=dataset_id,
        )

        return dataset_id

    finally:
        if temp_path.exists():
            temp_path.unlink()


async def _finalize_task_status(
    db: AsyncSession, task_id: UUID, dataset_ids: list[UUID]
) -> None:
    """Finalize the task status based on processing results.

    Args:
        db: Database session
        task_id: Synthesis task ID
        dataset_ids: List of successfully created dataset IDs
    """
    if not dataset_ids:
        await update_synthesis_task_status(
            db,
            task_id,
            TaskStatus.FAILED,
            error_msg="No valid files could be processed",
        )
        logger.error(
            "[BG Synthetic] Task failed: No valid files could be processed",
            taskId=task_id,
        )
    else:
        await update_synthesis_task_status(db, task_id, TaskStatus.DONE)

        logger.info(
            "[BG Synthetic] Task completed successfully",
            taskId=task_id,
            datasetCount=len(dataset_ids),
            datasetIds=dataset_ids,
        )
