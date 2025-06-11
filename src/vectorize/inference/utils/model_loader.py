# """Model loader for saved AI-Models with advanced optimizations."""

# import asyncio
# from functools import lru_cache
# from pathlib import Path
# from typing import Any

# import torch
# from loguru import logger
# from safetensors.torch import load_file
# from transformers import (
#     AutoConfig,
#     AutoModel,
#     AutoModelForMaskedLM,
#     AutoTokenizer,
#     T5EncoderModel,
# )

# from vectorize.ai_model.exceptions import ModelLoadError, ModelNotFoundError
# from vectorize.config import settings

# __all__ = [
#     "_clear_model_cache",
#     "_load_model",
#     "_preload_popular_models",
#     "get_cache_info",
# ]


# _DEVICE = torch.device(settings.inference_device)
# _IS_TORCH_NEW = torch.__version__ >= "2.7"
# _MODEL_NAME = "model.bin"

# # Base loading kwargs - only boolean/string values that are always supported
# _BASE_MODEL_KWARGS: dict[str, bool | str] = {
#     "trust_remote_code": True,
#     "low_cpu_mem_usage": True,
#     "use_safetensors": True,
# }


# def _get_optimized_kwargs() -> dict[str, Any]:
#     """Get device-optimized model loading parameters."""
#     # Start with base kwargs and add device-specific optimizations
#     kwargs: dict[str, Any] = _BASE_MODEL_KWARGS.copy()

#     if _DEVICE.type == "cuda":
#         # GPU optimizations
#         try:
#             # Only add these if accelerate is available
#             import accelerate

#             kwargs["torch_dtype"] = torch.float16  # Half precision for speed
#             kwargs["device_map"] = "auto"  # Automatic device placement
#         except ImportError:
#             logger.debug("accelerate not available, using basic CUDA settings")
#             kwargs["torch_dtype"] = torch.float16
#     else:
#         # CPU optimizations
#         kwargs["torch_dtype"] = torch.float32  # Full precision for CPU

#     return kwargs


# # Warm models cache for frequently accessed models
# _model_cache_stats: dict[str, int] = {}


# @lru_cache(maxsize=25)  # Increased cache size
# def _load_model(model_tag: str) -> tuple[torch.nn.Module, AutoTokenizer | None]:
#     """Load a Hugging Face model and its tokenizer with optimizations.

#     Supports loading from directories containing either a complete HF model snapshot
#     or just the config.json and model weights files. Includes warmup for better
#     first-request performance.

#     Args:
#         model_tag: Name of the model directory to load (without file extensions).

#     Returns:
#         A tuple containing:
#             - model: The loaded PyTorch model in evaluation mode and optimized
#             - tokenizer: The model's tokenizer if available, or None if not found

#     Raises:
#         ModelNotFoundError: If the model directory doesn't exist
#         ModelLoadError: If the model can't be successfully loaded
#     """
#     logger.debug(f"Loading model: {model_tag}")

#     # Timing setup for CUDA
#     start_time = None
#     end_time = None
#     if _DEVICE.type == "cuda" and torch.cuda.is_available():
#         start_time = torch.cuda.Event(enable_timing=True)
#         end_time = torch.cuda.Event(enable_timing=True)
#         start_time.record()

#     folder = Path(settings.model_inference_dir) / model_tag
#     if not folder.exists():
#         raise ModelNotFoundError(model_tag)

#     try:
#         cfg = AutoConfig.from_pretrained(folder)
#     except Exception as exc:
#         raise ModelLoadError(model_tag) from exc

#     # Clear GPU cache before loading if on CUDA
#     if _DEVICE.type == "cuda" and torch.cuda.is_available():
#         torch.cuda.empty_cache()

#     optimized_kwargs = _get_optimized_kwargs()

#     try:
#         model = _load_model_by_type(folder, cfg, optimized_kwargs)
#     except OSError:
#         model = _instantiate_from_weights(folder, cfg)
#     except Exception as exc:
#         raise ModelLoadError(model_tag) from exc

#     # Move to device and optimize
#     model = model.to(_DEVICE).eval().requires_grad_(False)

#     # Load tokenizer
#     tokenizer = _load_tokenizer(folder, model_tag)

#     # Warmup the model for better first-request performance
#     model = _warmup_model(model, tokenizer, model_tag)

#     # Log timing if available
#     if end_time and start_time:
#         end_time.record()
#         torch.cuda.synchronize()
#         load_time = start_time.elapsed_time(end_time)
#         logger.debug(f"Model {model_tag} loaded in {load_time:.2f}ms")

#     # Update cache stats
#     _model_cache_stats[model_tag] = _model_cache_stats.get(model_tag, 0) + 1

#     logger.info(
#         f"Loaded {model.__class__.__name__} on {_DEVICE}",
#         model_tag=model_tag,
#         has_tokenizer=tokenizer is not None,
#         cache_hits=_model_cache_stats[model_tag],
#     )

#     return model, tokenizer


# def _load_model_by_type(
#     folder: Path, cfg: AutoConfig, kwargs: dict[str, Any]
# ) -> torch.nn.Module:
#     """Load model based on its configuration type."""
#     # Use getattr for safe attribute access
#     model_type = getattr(cfg, "model_type", None)

#     if model_type == "t5":
#         return T5EncoderModel.from_pretrained(folder, **kwargs)

#     # Safe access to architectures
#     config_dict = getattr(cfg, "to_dict", dict)()
#     architectures = config_dict.get("architectures", [])

#     if architectures and "MaskedLM" in str(architectures[0]):
#         return AutoModelForMaskedLM.from_pretrained(folder, **kwargs)
#     return AutoModel.from_pretrained(folder, **kwargs)


# def _load_tokenizer(folder: Path, model_tag: str) -> AutoTokenizer | None:
#     """Load tokenizer with error handling."""
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(folder, trust_remote_code=True)

#         # Optimize tokenizer settings if methods are available
#         if hasattr(tokenizer, "enable_truncation"):
#             tokenizer.enable_truncation(max_length=512)
#         if hasattr(tokenizer, "enable_padding"):
#             tokenizer.enable_padding(pad_token="[PAD]")

#         return tokenizer
#     except Exception as e:
#         logger.warning(f"No tokenizer for '{model_tag}': {e}")
#         return None


# def _warmup_model(
#     model: torch.nn.Module, tokenizer: AutoTokenizer | None, model_tag: str
# ) -> torch.nn.Module:
#     """Perform model warmup for better first-request performance."""
#     try:
#         logger.debug(f"Warming up model: {model_tag}")

#         with torch.no_grad():
#             if tokenizer is not None:
#                 # Text-based warmup
#                 warmup_text = "warmup"
#                 inputs = tokenizer(
#                     warmup_text,
#                     return_tensors="pt",
#                     truncation=True,
#                     max_length=64,  # Short warmup
#                     padding=False,
#                 )
#                 inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}

#                 # Warmup forward pass
#                 _ = model(**inputs)
#             else:
#                 # Token-based warmup for models without tokenizer
#                 dummy_ids = torch.tensor([[1, 2, 3]], device=_DEVICE)
#                 dummy_mask = torch.ones_like(dummy_ids)
#                 _ = model(dummy_ids, attention_mask=dummy_mask)

#         logger.debug(f"Model warmup completed: {model_tag}")

#     except Exception as e:
#         logger.warning(f"Warmup failed for {model_tag}: {e}")
#         # Continue anyway - warmup failure shouldn't break loading

#     return model


# def _instantiate_from_weights(folder: Path, cfg: AutoConfig) -> torch.nn.Module:
#     """Create a model from config.json and load weights from a state dict file.

#     This is a fallback method for loading models that don't follow the standard
#     Hugging Face directory structure but contain the necessary components.

#     Args:
#         folder: Path to the directory containing config.json and weight files.
#         cfg: The AutoConfig object containing model configuration.

#     Returns:
#         The loaded PyTorch model in evaluation mode.

#     Raises:
#         ModelNotFoundError: If neither pytorch_model.bin nor model.bin exist
#     """
#     # Simplified approach: use AutoModel for everything when instantiating from weights
#     try:
#         model = AutoModel.from_config(cfg)
#         logger.debug("Created model using AutoModel.from_config")
#     except Exception as e:
#         logger.warning(f"Failed to create model from config: {e}")
#         raise ModelLoadError(f"Cannot instantiate model from config: {e}")

#     model.to(_DEVICE)

#     # Try safetensors first (faster and safer)
#     st_path = next(folder.glob("*.safetensors"), None)
#     if st_path:
#         logger.debug(f"Loading weights from safetensors: {st_path}")
#         try:
#             state = load_file(st_path, device=str(_DEVICE))
#             model.load_state_dict(state, strict=False)
#             return model.eval()
#         except Exception as e:
#             logger.warning(f"Failed to load safetensors: {e}")

#     # Fallback to pytorch_model.bin
#     f = folder / _MODEL_NAME
#     if f.is_file():
#         logger.debug(f"Loading weights from pytorch model: {f}")
#         try:
#             state = torch.load(
#                 f,
#                 mmap=_IS_TORCH_NEW,
#                 map_location=_DEVICE,
#                 weights_only=_IS_TORCH_NEW,
#             )
#             model.load_state_dict(state, strict=False)
#             return model.eval()
#         except Exception as e:
#             logger.warning(f"Failed to load pytorch model: {e}")

#     raise ModelNotFoundError(str(folder))


# async def _preload_popular_models() -> None:
#     """Preload frequently used models for better response times.

#     This should be called during application startup to warm up
#     commonly used models.
#     """
#     # Define popular models based on your usage patterns
#     popular_models = [
#         "pytorch_model",
#         "models--sentence-transformers--all-MiniLM-L6-v2",
#         # Add more based on your analytics
#     ]

#     logger.info("Starting model preloading...")

#     for model_tag in popular_models:
#         try:
#             # Run in thread pool to avoid blocking
#             await asyncio.get_event_loop().run_in_executor(None, _load_model, model_tag)
#             logger.info(f"✓ Preloaded model: {model_tag}")
#         except (ModelNotFoundError, ModelLoadError) as e:
#             logger.debug(f"Skipped preloading {model_tag}: {e}")
#         except Exception as e:
#             logger.warning(f"Failed to preload {model_tag}: {e}")

#     logger.info("Model preloading completed")


# def _clear_model_cache() -> None:
#     """Clear the model cache and free GPU memory."""
#     _load_model.cache_clear()
#     _model_cache_stats.clear()

#     if _DEVICE.type == "cuda" and torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         logger.info("Cleared model cache and GPU memory")
#     else:
#         logger.info("Cleared model cache")


# def get_cache_info() -> dict[str, Any]:
#     """Get information about the current model cache state."""
#     cache_info = _load_model.cache_info()

#     return {
#         "cache_hits": cache_info.hits,
#         "cache_misses": cache_info.misses,
#         "cache_size": cache_info.currsize,
#         "cache_maxsize": cache_info.maxsize,
#         "hit_rate": cache_info.hits / (cache_info.hits + cache_info.misses)
#         if (cache_info.hits + cache_info.misses) > 0
#         else 0,
#         "loaded_models": list(_model_cache_stats.keys()),
#         "model_usage": _model_cache_stats.copy(),
#     }
