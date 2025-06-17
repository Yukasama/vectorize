"""Model caching system that provides efficient management of loaded models."""

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Protocol

import torch
from loguru import logger
from transformers import AutoTokenizer

from .eviction import CacheEviction
from .usage_tracker import UsageTracker

__all__ = ["ModelCache"]


class ModelLoader(Protocol):
    """Protocol for model loader functions."""

    def __call__(
        self, model_tag: str
    ) -> tuple[torch.nn.Module, AutoTokenizer | None]: ...


class ModelCache:
    """Main class for model caching with usage tracking."""

    def __init__(
        self, max_models: int = 5, cache_file: str = "data/cache/model_usage_stats.json"
    ) -> None:
        """Initialize the model cache with usage tracking.

        Args:
            max_models: Maximum number of models to keep in the cache
            cache_file: Path to the file for storing model usage statistics
        """
        self.cache = OrderedDict()
        self.lock = threading.Lock()

        self.usage_tracker = UsageTracker(Path(cache_file))
        self.eviction = CacheEviction(self.usage_tracker, max_models)

        logger.info("Model cache initialized", max_models=max_models)

    def get(
        self, model_tag: str, loader_func: ModelLoader
    ) -> tuple[torch.nn.Module, AutoTokenizer | None]:
        """Get model from cache or load it if not cached."""
        with self.lock:
            self.usage_tracker.track_access(model_tag)

            # Cache hit?
            if model_tag in self.cache:
                self.cache.move_to_end(model_tag)
                logger.debug("Cache hit", model=model_tag)
                return self.cache[model_tag]

            # Eviction if necessary
            if self.eviction.should_evict(len(self.cache)):
                candidate = self.eviction.select_eviction_candidate(self.cache)
                if candidate:
                    self.eviction.evict_model(self.cache, candidate)

        # Load model (outside lock)
        logger.debug("Loading model", model=model_tag)
        model_data = loader_func(model_tag)

        # Cache update
        with self.lock:
            self.cache[model_tag] = model_data
            self.usage_tracker.save_stats()
            logger.info("Model cached", model=model_tag, cache_size=len(self.cache))

        return model_data

    def get_info(self) -> dict:
        """Get cache information for monitoring."""
        with self.lock:
            return {
                "cached_models": list(self.cache.keys()),
                "cache_size": len(self.cache),
                "max_size": self.eviction.max_models,
                "usage_stats": self.usage_tracker.get_stats(),
            }

    def clear(self) -> None:
        """Clear cache completely."""
        with self.lock:
            for model_tag, (model, _) in self.cache.items():
                try:
                    if hasattr(model, "cpu"):
                        model.cpu()
                except Exception as e:
                    logger.warning(
                        "Error during cache clear", model=model_tag, error=str(e)
                    )

            self.cache.clear()
            logger.info("Cache cleared")
