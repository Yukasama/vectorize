"""Core model loading functionality."""

from pathlib import Path
from typing import Any

import torch
from loguru import logger
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    T5EncoderModel,
)

from vectorize.ai_model.exceptions import ModelLoadError, ModelNotFoundError
from vectorize.config import settings

from .cache import model_cache_decorator, update_cache_stats
from .optimization import (
    clear_gpu_cache,
    get_device,
    get_optimized_kwargs,
    is_cuda_available,
)
from .warmup import warmup_model

__all__ = [
    "instantiate_from_weights",
    "load_model",
    "load_model_by_type",
    "load_tokenizer",
    "validate_model_directory",
]


_DEVICE = get_device()
_IS_TORCH_NEW = torch.__version__ >= "2.7"
_MODEL_NAME = "model.bin"


@model_cache_decorator(maxsize=25)
def load_model(model_tag: str) -> tuple[torch.nn.Module, AutoTokenizer | None]:
    """Load a Hugging Face model and its tokenizer with optimizations.

    Supports loading from directories containing either a complete HF model snapshot
    or just the config.json and model weights files. Includes warmup for better
    first-request performance.

    Args:
        model_tag: Name of the model directory to load (without file extensions).

    Returns:
        A tuple containing:
            - model: The loaded PyTorch model in evaluation mode and optimized
            - tokenizer: The model's tokenizer if available, or None if not found

    Raises:
        ModelNotFoundError: If the model directory doesn't exist
        ModelLoadError: If the model can't be successfully loaded
    """
    logger.debug(f"Loading model: {model_tag}")

    # Timing setup for CUDA
    start_time = None
    end_time = None
    if is_cuda_available():
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()

    folder = Path(settings.model_inference_dir) / model_tag
    if not folder.exists():
        raise ModelNotFoundError(model_tag)

    try:
        cfg = AutoConfig.from_pretrained(folder)
    except Exception as exc:
        raise ModelLoadError(model_tag) from exc

    # Clear GPU cache before loading if on CUDA
    clear_gpu_cache()

    optimized_kwargs = get_optimized_kwargs()

    try:
        model = load_model_by_type(folder, cfg, optimized_kwargs)
    except OSError:
        model = instantiate_from_weights(folder, cfg)
    except Exception as exc:
        raise ModelLoadError(model_tag) from exc

    # Move to device and optimize
    model = model.to(_DEVICE).eval().requires_grad_(False)

    # Load tokenizer
    tokenizer = load_tokenizer(folder, model_tag)

    # Warmup the model for better first-request performance
    model = warmup_model(model, tokenizer, model_tag)

    # Log timing if available
    if end_time and start_time:
        end_time.record()
        torch.cuda.synchronize()
        load_time = start_time.elapsed_time(end_time)
        logger.debug(f"Model {model_tag} loaded in {load_time:.2f}ms")

    # Update cache stats
    update_cache_stats(model_tag)

    logger.info(
        f"Loaded {model.__class__.__name__} on {_DEVICE}",
        model_tag=model_tag,
        has_tokenizer=tokenizer is not None,
    )

    return model, tokenizer


def load_model_by_type(
    folder: Path, cfg: AutoConfig, kwargs: dict[str, Any]
) -> torch.nn.Module:
    """Load model based on its configuration type.

    Args:
        folder: Path to the model directory.
        cfg: AutoConfig object with model configuration.
        kwargs: Loading arguments optimized for the current device.

    Returns:
        Loaded PyTorch model.
    """
    # Use getattr for safe attribute access
    model_type = getattr(cfg, "model_type", None)

    if model_type == "t5":
        logger.debug("Loading T5 encoder model")
        return T5EncoderModel.from_pretrained(folder, **kwargs)

    # Safe access to architectures
    config_dict = getattr(cfg, "to_dict", dict)()
    architectures = config_dict.get("architectures", [])

    if architectures and "MaskedLM" in str(architectures[0]):
        logger.debug("Loading MaskedLM model")
        return AutoModelForMaskedLM.from_pretrained(folder, **kwargs)

    logger.debug("Loading AutoModel")
    return AutoModel.from_pretrained(folder, **kwargs)


def load_tokenizer(folder: Path, model_tag: str) -> AutoTokenizer | None:
    """Load tokenizer with error handling and optimizations.

    Args:
        folder: Path to the model directory.
        model_tag: Model identifier for logging.

    Returns:
        Loaded tokenizer or None if not available.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(folder, trust_remote_code=True)

        # Optimize tokenizer settings if methods are available
        if hasattr(tokenizer, "enable_truncation"):
            tokenizer.enable_truncation(max_length=512)
        if hasattr(tokenizer, "enable_padding"):
            tokenizer.enable_padding(pad_token="[PAD]")

        logger.debug(f"Tokenizer loaded for {model_tag}")
        return tokenizer

    except Exception as e:
        logger.warning(f"No tokenizer for '{model_tag}': {e}")
        return None


def instantiate_from_weights(folder: Path, cfg: AutoConfig) -> torch.nn.Module:
    """Create a model from config.json and load weights from a state dict file.

    This is a fallback method for loading models that don't follow the standard
    Hugging Face directory structure but contain the necessary components.

    Args:
        folder: Path to the directory containing config.json and weight files.
        cfg: The AutoConfig object containing model configuration.

    Returns:
        The loaded PyTorch model in evaluation mode.

    Raises:
        ModelNotFoundError: If neither pytorch_model.bin nor model.bin exist
        ModelLoadError: If model instantiation fails
    """
    logger.debug("Using fallback: instantiating model from weights")

    # Simplified approach: use AutoModel for everything when instantiating from weights
    try:
        model = AutoModel.from_config(cfg)
        logger.debug("Created model using AutoModel.from_config")
    except Exception as e:
        logger.warning(f"Failed to create model from config: {e}")
        raise ModelLoadError(f"Cannot instantiate model from config: {e}")

    model.to(_DEVICE)

    # Try safetensors first (faster and safer)
    st_path = next(folder.glob("*.safetensors"), None)
    if st_path:
        logger.debug(f"Loading weights from safetensors: {st_path}")
        try:
            state = load_file(st_path, device=str(_DEVICE))
            model.load_state_dict(state, strict=False)
            return model.eval()
        except Exception as e:
            logger.warning(f"Failed to load safetensors: {e}")

    # Fallback to pytorch_model.bin
    f = folder / _MODEL_NAME
    if f.is_file():
        logger.debug(f"Loading weights from pytorch model: {f}")
        try:
            state = torch.load(
                f,
                mmap=_IS_TORCH_NEW,
                map_location=_DEVICE,
                weights_only=_IS_TORCH_NEW,
            )
            model.load_state_dict(state, strict=False)
            return model.eval()
        except Exception as e:
            logger.warning(f"Failed to load pytorch model: {e}")

    raise ModelNotFoundError(str(folder))


def validate_model_directory(folder: Path) -> dict[str, Any]:
    """Validate a model directory and return information about its contents.

    Args:
        folder: Path to the model directory to validate.

    Returns:
        Dictionary with validation results and file information.
    """
    validation_result = {
        "path": str(folder),
        "exists": folder.exists(),
        "is_directory": folder.is_dir() if folder.exists() else False,
        "files": [],
        "has_config": False,
        "has_weights": False,
        "has_tokenizer": False,
        "weight_format": None,
    }

    if not validation_result["exists"]:
        return validation_result

    if not validation_result["is_directory"]:
        validation_result["error"] = "Path exists but is not a directory"
        return validation_result

    # Check for essential files
    for file_path in folder.iterdir():
        if file_path.is_file():
            filename = file_path.name
            validation_result["files"].append(filename)

            if filename == "config.json":
                validation_result["has_config"] = True
            elif filename.endswith(".safetensors"):
                validation_result["has_weights"] = True
                validation_result["weight_format"] = "safetensors"
            elif filename in ["pytorch_model.bin", "model.bin"]:
                validation_result["has_weights"] = True
                if not validation_result["weight_format"]:  # Prefer safetensors
                    validation_result["weight_format"] = "pytorch"
            elif filename in ["tokenizer.json", "tokenizer_config.json"]:
                validation_result["has_tokenizer"] = True

    # Overall validation status
    validation_result["is_valid"] = (
        validation_result["has_config"] and validation_result["has_weights"]
    )

    return validation_result
