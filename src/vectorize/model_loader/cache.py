"""Model cache management and statistics."""

from functools import lru_cache
from typing import Any

from loguru import logger

from .optimization import clear_gpu_cache

__all__ = [
    "clear_model_cache",
    "get_cache_info",
    "get_cache_stats",
    "model_cache_decorator",
    "update_cache_stats",
]


# Cache statistics tracking
_model_cache_stats: dict[str, int] = {}


def model_cache_decorator(maxsize: int = 25):
    """Decorator for caching model loading with statistics tracking.

    Args:
        maxsize: Maximum number of models to cache.

    Returns:
        LRU cache decorator with integrated statistics.
    """
    return lru_cache(maxsize=maxsize)


def update_cache_stats(model_tag: str) -> None:
    """Update cache usage statistics for a model.

    Args:
        model_tag: The model identifier to track.
    """
    _model_cache_stats[model_tag] = _model_cache_stats.get(model_tag, 0) + 1


def get_cache_stats() -> dict[str, int]:
    """Get model usage statistics.

    Returns:
        Dictionary mapping model tags to usage counts.
    """
    return _model_cache_stats.copy()


def clear_model_cache(cache_function) -> None:
    """Clear the model cache and free GPU memory.

    Args:
        cache_function: The cached function to clear (typically _load_model).
    """
    if hasattr(cache_function, "cache_clear"):
        cache_function.cache_clear()

    _model_cache_stats.clear()
    clear_gpu_cache()

    logger.info("Model cache cleared successfully")


def get_cache_info(cache_function) -> dict[str, Any]:
    """Get comprehensive information about the current model cache state.

    Args:
        cache_function: The cached function to inspect.

    Returns:
        Dictionary with cache statistics and model usage information.
    """
    if not hasattr(cache_function, "cache_info"):
        return {
            "error": "Cache function does not support cache_info",
            "loaded_models": list(_model_cache_stats.keys()),
            "model_usage": _model_cache_stats.copy(),
        }

    cache_info = cache_function.cache_info()

    return {
        "cache_hits": cache_info.hits,
        "cache_misses": cache_info.misses,
        "cache_size": cache_info.currsize,
        "cache_maxsize": cache_info.maxsize,
        "hit_rate": cache_info.hits / (cache_info.hits + cache_info.misses)
        if (cache_info.hits + cache_info.misses) > 0
        else 0,
        "loaded_models": list(_model_cache_stats.keys()),
        "model_usage": _model_cache_stats.copy(),
    }


def get_cache_efficiency() -> dict[str, Any]:
    """Calculate cache efficiency metrics.

    Returns:
        Dictionary with efficiency analysis.
    """
    if not _model_cache_stats:
        return {"status": "no_models_loaded"}

    total_requests = sum(_model_cache_stats.values())
    unique_models = len(_model_cache_stats)

    # Most and least used models
    most_used = max(_model_cache_stats.items(), key=lambda x: x[1])
    least_used = min(_model_cache_stats.items(), key=lambda x: x[1])

    return {
        "total_requests": total_requests,
        "unique_models_loaded": unique_models,
        "average_requests_per_model": total_requests / unique_models,
        "most_used_model": {"name": most_used[0], "requests": most_used[1]},
        "least_used_model": {"name": least_used[0], "requests": least_used[1]},
        "cache_effectiveness": most_used[1]
        / total_requests
        * 100,  # % of requests to most popular model
    }


def suggest_cache_optimization() -> dict[str, Any]:
    """Analyze cache usage and suggest optimizations.

    Returns:
        Dictionary with optimization suggestions.
    """
    efficiency = get_cache_efficiency()

    if efficiency.get("status") == "no_models_loaded":
        return {"suggestion": "No models loaded yet, no optimization needed"}

    suggestions = []

    # Check if cache is underutilized
    if efficiency["unique_models_loaded"] < 5:
        suggestions.append("Consider preloading more frequently used models")

    # Check for uneven usage
    if efficiency["cache_effectiveness"] > 80:
        suggestions.append(
            f"Cache is highly effective - {efficiency['most_used_model']['name']} dominates usage"
        )
    elif efficiency["cache_effectiveness"] < 20:
        suggestions.append(
            "Usage is well distributed across models - good cache utilization"
        )

    # Check for low-usage models
    low_usage_models = [
        model
        for model, count in _model_cache_stats.items()
        if count < efficiency["average_requests_per_model"] * 0.1
    ]

    if low_usage_models:
        suggestions.append(f"Consider removing rarely used models: {low_usage_models}")

    return {
        "efficiency_score": min(100, efficiency["cache_effectiveness"]),
        "suggestions": suggestions,
        "metrics": efficiency,
    }
