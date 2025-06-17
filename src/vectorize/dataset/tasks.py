"""Task to upload Hugging Face datasets to the database."""

from uuid import UUID

import dramatiq
from datasets import load_dataset_builder
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.config.db import engine
from vectorize.dataset.utils.process_hf_model import _process_dataset

__all__ = ["upload_hf_dataset_bg"]


@dramatiq.actor(max_retries=1)
async def upload_hf_dataset_bg(
    dataset_tag: str, task_id: str, subsets: list[str]
) -> None:
    """Upload a Hugging Face dataset in the background.

    Processes all subsets and splits of a Hugging Face dataset, downloading
    and converting them to JSONL format for storage in the database.
    Updates the task status upon completion.

    Args:
        db: Database session for persistence operations.
        dataset_tag: Tag identifier for the Hugging Face dataset (e.g., 'squad').
        task_id: UUID of the upload task to track progress.
        subsets: List of dataset subsets to process (e.g., ['easy', 'hard']).

    Raises:
        Exception: If dataset loading or processing fails.
    """
    async with AsyncSession(engine, expire_on_commit=False) as db:
        try:
            for subset in subsets:
                info = load_dataset_builder(dataset_tag, name=subset).info
                logger.debug(
                    "Processing Hugging Face dataset",
                    dataset_tag=dataset_tag,
                    subset=subset,
                    splits=list(info.splits.keys()) if info.splits else None,
                    features=list(info.features.keys()) if info.features else None,
                )
                await _process_dataset(db, dataset_tag, UUID(task_id), subset, info)

            await db.commit()
            logger.info(
                "HF Dataset upload complete", dataset_tag=dataset_tag, task_id=task_id
            )

        except Exception as e:
            await db.rollback()
            logger.error(
                "Error in HF dataset background task",
                dataset_tag=dataset_tag,
                task_id=task_id,
                error=str(e),
                exc_info=True,
            )
            raise
