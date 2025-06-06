"""Process Hugging Face dataset splits and subsets."""

import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from uuid import UUID

import orjson
from datasets import Dataset as HFDataset
from datasets import DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from datasets.info import DatasetInfo
from loguru import logger
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel.ext.asyncio.session import AsyncSession
from tqdm.auto import tqdm

from vectorize.common.task_status import TaskStatus
from vectorize.config.config import settings

from ..classification import Classification
from ..dataset_source import DatasetSource
from ..models import Dataset
from ..repository import update_upload_task_status, upload_dataset_db

__all__ = ["_get_dataset_rows", "_process_dataset"]


async def _process_dataset(
    db: AsyncSession,
    dataset_tag: str,
    task_id: UUID,
    subset: str,
    info: DatasetInfo,
) -> None:
    """Process splits for a single dataset subset."""
    if subset == "default":
        if not info.splits:
            await _process_single_dataset(db, dataset_tag, task_id)
        else:
            for split in info.splits:
                await _process_single_dataset(db, dataset_tag, task_id, split)
    elif not info.splits:
        await _process_single_dataset(db, dataset_tag, task_id, subset=subset)
    else:
        for split in info.splits:
            await _process_single_dataset(db, dataset_tag, task_id, split, subset)


async def _process_single_dataset(
    db: AsyncSession,
    dataset_tag: str,
    task_id: UUID,
    split: str | None = None,
    subset: str | None = "default",
) -> None:
    """Process a single dataset split/subset and save to database.

    Downloads a specific split and subset of a Hugging Face dataset,
    converts it to JSONL format, and creates a database record.

    Args:
        db: Database session for persistence operations.
        dataset_tag: Tag identifier for the Hugging Face dataset.
        task_id: UUID of the upload task for status tracking.
        split: Name of the dataset split (e.g., 'train', 'validation').
        subset: Name of the dataset subset/configuration. Defaults to "default".

    Raises:
        Exception: If dataset loading, file writing, or database operations fail.
    """
    file_path = None

    try:
        ds = load_dataset(dataset_tag, name=subset, split=split, streaming=False)
        logger.debug(
            "Loaded HF dataset", dataset_tag=dataset_tag, split=split, subset=subset
        )

        parts = [dataset_tag.replace("/", "_")]
        if split:
            parts.append(split)
        if subset and subset != "default":
            parts.append(subset)

        file_name = f"{'_'.join(parts)}.jsonl"
        file_path = settings.dataset_upload_dir / file_name
        _write_jsonl(ds, file_path)

        dataset_name = dataset_tag
        if split:
            dataset_name += f"_{split}"
        if subset and subset != "default":
            dataset_name += f"_{subset}"

        dataset = Dataset(
            name=dataset_name,
            classification=Classification.SENTENCE_TRIPLES,
            file_name=file_name,
            source=DatasetSource.HUGGINGFACE,
            rows=_get_dataset_rows(ds),
        )

        dataset_id = await upload_dataset_db(db, dataset)
        await update_upload_task_status(db, task_id, TaskStatus.DONE)
        logger.debug("HF Dataset saved", dataset_tag=dataset_tag, dataset_id=dataset_id)

    except SQLAlchemyError as e:
        if file_path and file_path.exists():
            file_path.unlink()
            logger.debug("Cleaned up file after database error", file_path=file_path)

        await update_upload_task_status(
            db, task_id, TaskStatus.FAILED, error_msg=f"DB insert failed: {e!s}"
        )
        raise

    except Exception as e:
        await update_upload_task_status(
            db,
            task_id,
            TaskStatus.FAILED,
            error_msg=f"Dataset processing failed: {e!s}",
        )
        logger.error(
            "Error processing Hugging Face dataset",
            dataset_tag=dataset_tag,
            split=split,
            subset=subset,
            error=str(e),
        )
        raise


_LINE_SEP = b"\xe2\x80\xa8"
_PARA_SEP = b"\xe2\x80\xa9"


def _write_jsonl(
    ds: HFDataset | Iterable[dict],
    path: Path,
    batch_size: int = 2048,
) -> int:
    if isinstance(ds, HFDataset):
        ds = ds.with_format("python")
    tmp_fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(tmp_fd)
    row_count, buf = 0, bytearray()

    with (
        path.parent.joinpath(tmp_name).open("wb") as fh,
        tqdm(total=getattr(ds, "num_rows", None), unit="ex", desc=path.name) as bar,
    ):
        for example in ds:
            try:
                rec = orjson.dumps(example)
            except orjson.JSONEncodeError:
                logger.warning("Skipped unserialisable row #%d", row_count)
                continue
            if (_LINE_SEP in rec) or (_PARA_SEP in rec):
                rec = rec.replace(_LINE_SEP, b"\\u2028").replace(_PARA_SEP, b"\\u2029")
            buf += rec + b"\n"
            row_count += 1
            if row_count % batch_size == 0:
                fh.write(buf)
                buf.clear()
                bar.update(batch_size)
        if buf:
            fh.write(buf)
            bar.update(len(buf))

    Path.replace(path.parent / tmp_name, path)
    return row_count


def _get_dataset_rows(
    ds: HFDataset | DatasetDict | IterableDataset | IterableDatasetDict,
) -> int:
    """Get dataset row count safely, handling different dataset types."""
    if isinstance(ds, HFDataset):
        return ds.num_rows
    if isinstance(ds, DatasetDict):
        return sum(split.num_rows for split in ds.values())
    if isinstance(ds, (IterableDataset, IterableDatasetDict)):
        return 0
