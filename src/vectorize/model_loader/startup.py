"""Model loader startup and shutdown functionality."""

from loguru import logger

__all__ = ["cleanup_models_on_shutdown", "preload_models_on_startup"]


async def preload_models_on_startup() -> None:
    """Preload popular models for better performance during startup."""
    try:
        from . import load_model, preload_popular_models

        # Diese Modelle existieren laut Ihrer seed.py
        popular_models = ["pytorch_model", "big_model", "huge_model"]
        results = await preload_popular_models(load_model, popular_models)

        successful = sum(1 for status in results.values() if status == "success")
        if successful > 0:
            logger.info(
                f"Model preloading: {successful}/{len(popular_models)} models loaded"
            )

    except Exception as e:
        logger.debug(f"Model preloading skipped: {e}")


async def cleanup_models_on_shutdown() -> None:
    """Clean up model cache and free resources on shutdown."""
    try:
        from . import clear_model_cache, load_model

        clear_model_cache(load_model)
        logger.debug("Model cache cleared on shutdown")

    except Exception as e:
        logger.debug(f"Model cache cleanup failed: {e}")
