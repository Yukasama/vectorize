"""Model loader for saved AI-Models."""

from functools import lru_cache
from pathlib import Path
from typing import Any, cast

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

__all__ = ["_load_model"]


_DEVICE = torch.device(settings.inference_device)
_IS_TORCH_NEW = torch.__version__ >= "2.7"
_MODEL_NAME = "model.bin"

_COMMON_MODEL_KWARGS = {
    "trust_remote_code": True,
    "torch_dtype": "auto",
    "low_cpu_mem_usage": True,
    "use_safetensors": True,
}


@lru_cache(maxsize=10)
def _load_model(model_tag: str) -> tuple[torch.nn.Module, AutoTokenizer | None]:
    """Load a Hugging Face model and its tokenizer from a checkpoint directory.

    Supports loading from directories containing either a complete HF model snapshot
    or just the config.json and model weights files.

    Args:
        model_tag: Name of the model directory to load (without file extensions).

    Returns:
        A tuple containing:
            - model: The loaded PyTorch model in evaluation mode on CPU
            - tokenizer: The model's tokenizer if available, or None if not found

    Raises:
        ModelNotFoundError: If the model directory doesn't exist
        ModelLoadError: If the model can't be successfully loaded
    """
    folder = Path(settings.model_inference_dir) / model_tag
    if not folder.exists():
        raise ModelNotFoundError(model_tag)

    try:
        cfg = AutoConfig.from_pretrained(folder)
    except Exception as exc:
        raise ModelLoadError(model_tag) from exc

    try:
        if cfg.model_type == "t5":
            model = T5EncoderModel.from_pretrained(folder, **_COMMON_MODEL_KWARGS)
        elif (
            "architectures" in cfg.to_dict()
            and cfg.architectures
            and "MaskedLM" in cfg.architectures[0]
        ):
            model = AutoModelForMaskedLM.from_pretrained(folder, **_COMMON_MODEL_KWARGS)
        else:
            model = AutoModel.from_pretrained(folder, **_COMMON_MODEL_KWARGS)

    except OSError:
        model = _instantiate_from_weights(folder, cfg) # type: ignore
    except Exception as exc:
        raise ModelLoadError(model_tag) from exc

    model = model.to(_DEVICE).eval().requires_grad_(False)  # type: ignore

    try:
        tok = AutoTokenizer.from_pretrained(folder, trust_remote_code=True)
    except Exception as e:
        logger.warning("No tokenizer for '{}': {}", model_tag, e)
        tok = None

    logger.debug(
        "Loaded {} on {}",
        model.__class__.__name__,
        _DEVICE,
        tokenizer=tok is not None,
    )
    return model, tok


def _instantiate_from_weights(folder: Path, cfg: AutoConfig) -> torch.nn.Module:
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
    """
    if cast(Any, cfg) == "t5":
        model = T5EncoderModel.from_config(cfg)  # type: ignore
    elif (
        "architectures" in cfg.to_dict()  # type: ignore
        and cfg.architectures  # type: ignore
        and "MaskedLM" in cfg.architectures[0]  # type: ignore
    ):
        model = AutoModelForMaskedLM.from_config(cfg)
    else:
        model = AutoModel.from_config(cfg)
    model.to(_DEVICE)

    st_path = next(folder.glob("*.safetensors"), None)
    if st_path:
        logger.debug("Loading weights from {}", st_path)
        state = load_file(st_path, device=str(_DEVICE))
        model.load_state_dict(state, strict=False)
        return model.eval()

    f = folder / _MODEL_NAME
    if f.is_file():
        state = torch.load(
            f,
            mmap=_IS_TORCH_NEW,
            map_location=_DEVICE,
            weights_only=_IS_TORCH_NEW,
        )
        model.load_state_dict(state, strict=False)
        return model.eval()

    raise ModelNotFoundError(str(folder))
