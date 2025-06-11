"""Device-specific optimizations for model loading."""

from typing import Any

import torch
from loguru import logger

from vectorize.config import settings

__all__ = ["clear_gpu_cache", "get_device", "get_optimized_kwargs"]


_DEVICE = torch.device(settings.inference_device)

# Base loading kwargs - only boolean/string values that are always supported
_BASE_MODEL_KWARGS: dict[str, bool | str] = {
    "trust_remote_code": True,
    "low_cpu_mem_usage": True,
    "use_safetensors": True,
}


def get_device() -> torch.device:
    """Get the configured device for inference."""
    return _DEVICE


def get_optimized_kwargs() -> dict[str, Any]:
    """Get device-optimized model loading parameters.

    Returns:
        Dictionary with optimized kwargs for the current device.
    """
    # Start with base kwargs and add device-specific optimizations
    kwargs: dict[str, Any] = _BASE_MODEL_KWARGS.copy()

    if _DEVICE.type == "cuda":
        # GPU optimizations
        try:
            # Only add these if accelerate is available
            import accelerate  # noqa

            kwargs["torch_dtype"] = torch.float16  # Half precision for speed
            kwargs["device_map"] = "auto"  # Automatic device placement
            logger.debug("Using accelerate optimizations for CUDA")
        except ImportError:
            logger.debug("accelerate not available, using basic CUDA settings")
            kwargs["torch_dtype"] = torch.float16
    else:
        # CPU optimizations
        kwargs["torch_dtype"] = torch.float32  # Full precision for CPU

    return kwargs


def clear_gpu_cache() -> None:
    """Clear GPU cache if on CUDA device."""
    if _DEVICE.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("GPU cache cleared")


def is_cuda_available() -> bool:
    """Check if CUDA is available and device is set to CUDA."""
    return _DEVICE.type == "cuda" and torch.cuda.is_available()


def get_memory_info() -> dict[str, Any]:
    """Get memory information for the current device.

    Returns:
        Dictionary with memory statistics or empty dict for CPU.
    """
    if not is_cuda_available():
        return {}

    try:
        total_memory = torch.cuda.get_device_properties(_DEVICE).total_memory
        allocated = torch.cuda.memory_allocated(_DEVICE)
        cached = torch.cuda.memory_reserved(_DEVICE)

        return {
            "total_memory_gb": total_memory / (1024**3),
            "allocated_gb": allocated / (1024**3),
            "cached_gb": cached / (1024**3),
            "free_gb": (total_memory - allocated) / (1024**3),
            "utilization_percent": (allocated / total_memory) * 100,
        }
    except Exception as e:
        logger.warning(f"Failed to get GPU memory info: {e}")
        return {}
