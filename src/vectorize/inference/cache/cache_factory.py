"""Factory module for creating model caches based on configuration settings."""

from loguru import logger

from vectorize.config import settings

from .model_cache import ModelCache  # Original Cache
from .vram_model_cache import VRAMModelCache  # VRAM Cache

__all__ = ["create_model_cache"]


def create_model_cache(
    cache_file: str = "data/cache/model_usage_stats.json",
) -> ModelCache | VRAMModelCache:
    """Factory-Funktion für Cache basierend auf Konfiguration."""
    if settings.cache_strategy == "fixed_size":
        logger.info(
            "Creating fixed-size model cache", max_models=settings.cache_max_models
        )

        return ModelCache(max_models=settings.cache_max_models, cache_file=cache_file)

    if settings.cache_strategy == "vram_aware":
        logger.info(
            "Creating VRAM-aware model cache",
            safety_margin_gb=settings.cache_vram_safety_margin_gb,
        )

        return VRAMModelCache(
            safety_margin_gb=settings.cache_vram_safety_margin_gb, cache_file=cache_file
        )

    logger.error("Unknown cache strategy", strategy=settings.cache_strategy)
    # Fallback zu Fixed-Size
    logger.warning("Falling back to fixed_size strategy")
    return ModelCache(max_models=5, cache_file=cache_file)
