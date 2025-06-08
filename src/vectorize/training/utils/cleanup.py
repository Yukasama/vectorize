"""Resource cleanup utilities for SBERT training."""

import gc

import torch
from loguru import logger


def cleanup_resources(model: object = None) -> None:
    """Cleans up model and frees memory resources.

    Args:
        model: The model object to delete (optional).
    """
    if model is not None:
        try:
            del model
        except Exception as exc:
            logger.warning("Cleanup failed (model): {}", str(exc))
    try:
        gc.collect()
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        logger.warning("Cleanup failed (GC/CUDA): {}", str(exc))
