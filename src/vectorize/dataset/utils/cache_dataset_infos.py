"""Caching utilities for Hugging Face dataset operations."""

import time

from datasets import get_dataset_infos
from datasets.info import DatasetInfo
from loguru import logger

_dataset_info_cache: dict[str, tuple[dict[str, DatasetInfo], float]] = {}
_CACHE_TTL = 3600

__all__ = ["_get_cached_dataset_infos"]


def _get_cached_dataset_infos(dataset_tag: str) -> dict[str, DatasetInfo]:
    """Get dataset infos with caching to improve performance.

    Args:
        dataset_tag: The Hugging Face dataset tag

    Returns:
        Dictionary mapping subset names to DatasetInfo objects
    """
    current_time = time.time()

    if dataset_tag in _dataset_info_cache:
        cached_infos, cache_time = _dataset_info_cache[dataset_tag]
        if current_time - cache_time < _CACHE_TTL:
            logger.debug("Using cached dataset info", dataset_tag=dataset_tag)
            return cached_infos

    logger.debug("Fetching dataset info from Hugging Face", dataset_tag=dataset_tag)
    dataset_infos = get_dataset_infos(dataset_tag)

    _dataset_info_cache[dataset_tag] = (dataset_infos, current_time)
    return dataset_infos
