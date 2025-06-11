"""Model loader package for optimized AI model loading and caching."""

from .cache import (
    clear_model_cache,
    get_cache_efficiency,
    get_cache_info,
    get_cache_stats,
    suggest_cache_optimization,
)
from .core import (
    instantiate_from_weights,
    load_model,
    load_model_by_type,
    load_tokenizer,
    validate_model_directory,
)
from .optimization import (
    clear_gpu_cache,
    get_device,
    get_memory_info,
    get_optimized_kwargs,
    is_cuda_available,
)
from .startup import cleanup_models_on_shutdown, preload_models_on_startup  # Neu
from .warmup import (
    benchmark_warmup_impact,
    preload_popular_models,
    warmup_model,
)

# Main interface
__all__ = [
    "benchmark_warmup_impact",
    "cleanup_models_on_shutdown",
    "clear_gpu_cache",
    "clear_model_cache",
    "get_cache_efficiency",
    "get_cache_info",
    "get_cache_stats",
    "get_device",
    "get_memory_info",
    "get_optimized_kwargs",
    "instantiate_from_weights",
    "is_cuda_available",
    "load_model",
    "load_model_by_type",
    "load_tokenizer",
    "preload_models_on_startup",
    "preload_popular_models",
    "suggest_cache_optimization",
    "validate_model_directory",
    "warmup_model",
]
