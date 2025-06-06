"""Task to upload Hugging Face datasets to the database."""

from uuid import UUID

from datasets.info import DatasetInfo
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.dataset.utils.process_hf_model import _process_dataset

__all__ = ["upload_hf_dataset_bg"]


async def upload_hf_dataset_bg(
    db: AsyncSession,
    dataset_tag: str,
    task_id: UUID,
    dataset_infos: dict[str, DatasetInfo],
) -> None:
    """Upload a Hugging Face dataset in the background.

    Processes all subsets and splits of a Hugging Face dataset, downloading
    and converting them to JSONL format for storage in the database.
    Updates the task status upon completion.

    Args:
        db: Database session for persistence operations.
        dataset_tag: Tag identifier for the Hugging Face dataset (e.g., 'squad').
        task_id: UUID of the upload task to track progress.
        dataset_infos: DatasetInfo objects from HuggingFace.

    Raises:
        Exception: If dataset loading or processing fails.
    """
    for subset, info in dataset_infos.items():
        logger.debug(
            "Processing Hugging Face dataset",
            dataset_tag=dataset_tag,
            subset=subset,
            splits=list(info.splits.keys()) if info.splits else None,
            features=list(info.features.keys()) if info.features else None,
        )

        await _process_dataset(db, dataset_tag, task_id, subset, info)

    logger.info("HF Dataset upload complete", dataset_tag=dataset_tag, task_id=task_id)
