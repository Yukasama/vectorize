"""Model warmup functionality for better first-request performance."""

import asyncio

import torch
from loguru import logger
from transformers import AutoTokenizer

from vectorize.ai_model.exceptions import ModelLoadError, ModelNotFoundError

from .optimization import get_device

__all__ = ["benchmark_warmup_impact", "preload_popular_models", "warmup_model"]


_DEVICE = get_device()


def warmup_model(
    model: torch.nn.Module, tokenizer: AutoTokenizer | None, model_tag: str
) -> torch.nn.Module:
    """Perform model warmup for better first-request performance.

    Args:
        model: The PyTorch model to warm up.
        tokenizer: The tokenizer for the model (can be None).
        model_tag: Identifier for logging purposes.

    Returns:
        The warmed-up model (same instance, but with initialized CUDA kernels).
    """
    try:
        logger.debug(f"Warming up model: {model_tag}")

        with torch.no_grad():
            if tokenizer is not None:
                _warmup_with_tokenizer(model, tokenizer)
            else:
                _warmup_without_tokenizer(model)

        logger.debug(f"Model warmup completed: {model_tag}")

    except Exception as e:
        logger.warning(f"Warmup failed for {model_tag}: {e}")
        # Continue anyway - warmup failure shouldn't break loading

    return model


def _warmup_with_tokenizer(model: torch.nn.Module, tokenizer: AutoTokenizer) -> None:
    """Warmup model using tokenizer for text input.

    Args:
        model: The model to warm up.
        tokenizer: The tokenizer to use for encoding.
    """
    warmup_text = "warmup"
    inputs = tokenizer(
        warmup_text,
        return_tensors="pt",
        truncation=True,
        max_length=64,  # Short warmup to minimize overhead
        padding=False,
    )
    inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}

    # Warmup forward pass
    _ = model(**inputs)


def _warmup_without_tokenizer(model: torch.nn.Module) -> None:
    """Warmup model using dummy token IDs (for models without tokenizer).

    Args:
        model: The model to warm up.
    """
    dummy_ids = torch.tensor([[1, 2, 3]], device=_DEVICE)
    dummy_mask = torch.ones_like(dummy_ids)
    _ = model(dummy_ids, attention_mask=dummy_mask)


async def preload_popular_models(
    model_loader_function, popular_models: list[str] | None = None
) -> dict[str, str]:
    """Preload frequently used models for better response times.

    This should be called during application startup to warm up
    commonly used models.

    Args:
        model_loader_function: The function to use for loading models.
        popular_models: List of model tags to preload. If None, uses defaults.

    Returns:
        Dictionary with preload results for each model.
    """
    if popular_models is None:
        popular_models = [
            "pytorch_model",
            "models--sentence-transformers--all-MiniLM-L6-v2",
            # Add more based on your analytics
        ]

    logger.info("Starting model preloading...")
    results = {}

    for model_tag in popular_models:
        try:
            # Run in thread pool to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                None, model_loader_function, model_tag
            )
            logger.info(f"✓ Preloaded model: {model_tag}")
            results[model_tag] = "success"

        except (ModelNotFoundError, ModelLoadError) as e:
            logger.debug(f"Skipped preloading {model_tag}: {e}")
            results[model_tag] = f"skipped: {e!s}"

        except Exception as e:
            logger.warning(f"Failed to preload {model_tag}: {e}")
            results[model_tag] = f"failed: {e!s}"

    logger.info("Model preloading completed")
    return results


def benchmark_warmup_impact(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer | None,
    model_tag: str,
    test_inputs: list[str] | None = None,
) -> dict[str, float]:
    """Benchmark the impact of warmup on inference speed.

    Args:
        model: The model to benchmark.
        tokenizer: The tokenizer for the model.
        model_tag: Model identifier for logging.
        test_inputs: Test inputs to use for benchmarking.

    Returns:
        Dictionary with timing results in milliseconds.
    """
    if test_inputs is None:
        test_inputs = ["test sentence", "another test", "benchmark text"]

    # Measure cold start (without warmup)
    if _DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Cold start timing
    start_time = (
        torch.cuda.Event(enable_timing=True) if _DEVICE.type == "cuda" else None
    )
    end_time = torch.cuda.Event(enable_timing=True) if _DEVICE.type == "cuda" else None

    if start_time:
        start_time.record()

    # Run inference without warmup
    with torch.no_grad():
        for text in test_inputs:
            if tokenizer:
                inputs = tokenizer(text, return_tensors="pt")
                inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}
                _ = model(**inputs)

    if end_time and start_time:
        end_time.record()
        torch.cuda.synchronize()
        cold_time = start_time.elapsed_time(end_time)
    else:
        cold_time = 0.0

    # Now with warmup
    warmup_model(model, tokenizer, model_tag)

    if start_time:
        start_time.record()

    with torch.no_grad():
        for text in test_inputs:
            if tokenizer:
                inputs = tokenizer(text, return_tensors="pt")
                inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}
                _ = model(**inputs)

    if end_time and start_time:
        end_time.record()
        torch.cuda.synchronize()
        warm_time = start_time.elapsed_time(end_time)
    else:
        warm_time = 0.0

    improvement = ((cold_time - warm_time) / cold_time * 100) if cold_time > 0 else 0

    return {
        "cold_start_ms": cold_time,
        "after_warmup_ms": warm_time,
        "improvement_percent": improvement,
        "test_inputs_count": len(test_inputs),
    }
