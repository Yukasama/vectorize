"""Load a model and optional tokenizer."""

from functools import cache
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer

from txt2vec.ai_model.exceptions import ModelLoadError, ModelNotFoundError
from txt2vec.config import settings

__all__ = ["load_model"]


_DEVICE = torch.device("cpu")


@cache
def load_model(model_tag: str) -> tuple[torch.nn.Module, AutoTokenizer | None]:
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
    folder = Path(settings.model_upload_dir) / model_tag
    logger.debug("Loading model '{}' from {}", model_tag, folder)

    if not folder.exists():
        raise ModelNotFoundError(model_tag)

    model = None
    try:
        config = AutoConfig.from_pretrained(folder)
        if hasattr(config, "architectures") and config.architectures:
            arch = config.architectures[0]
            logger.debug(f"Loading model with architecture {arch}")

            if "MaskedLM" in arch:
                model = (
                    AutoModelForMaskedLM.from_pretrained(folder, trust_remote_code=True)
                    .to(_DEVICE)
                    .eval()
                )
            else:
                model = (
                    AutoModel.from_pretrained(folder, trust_remote_code=True)
                    .to(_DEVICE)
                    .eval()
                )
        else:
            model = (
                AutoModel.from_pretrained(folder, trust_remote_code=True)
                .to(_DEVICE)
                .eval()
            )
    except (FileNotFoundError, OSError):
        try:
            model = _instantiate_from_weights(folder)
        except Exception as exc:
            raise ModelLoadError(model_tag) from exc
    except Exception as exc:
        raise ModelLoadError(model_tag) from exc

    if model is None:
        raise ModelLoadError(model_tag)

    try:
        tokenizer = AutoTokenizer.from_pretrained(folder)
    except Exception as e:
        tokenizer = None
        logger.warning("No tokenizer found for model '{}': {}", model_tag, str(e))

    return model, tokenizer


def _instantiate_from_weights(folder: Path) -> torch.nn.Module:
    """Create a model from config.json and load weights from a state dict file.

    This is a fallback method for loading models that don't follow the standard
    Hugging Face directory structure but contain the necessary components.

    Args:
        folder: Path to the directory containing config.json and weight files.

    Returns:
        The loaded PyTorch model in evaluation mode.

    Raises:
        ModelNotFoundError: If neither pytorch_model.bin nor model.bin exist
    """
    cfg = AutoConfig.from_pretrained(folder)

    if hasattr(cfg, "architectures") and cfg.architectures:
        architecture = cfg.architectures[0]
        logger.debug(f"Using architecture from config: {architecture}")

        if "MaskedLM" in architecture:
            model = AutoModelForMaskedLM.from_config(cfg).to(_DEVICE)
        else:
            model = AutoModel.from_config(cfg).to(_DEVICE)
    else:
        model = AutoModel.from_config(cfg).to(_DEVICE)

    for fname in ("pytorch_model.bin", "model.bin"):
        weight_file = folder / fname
        if weight_file.is_file():
            logger.debug("Loading state-dict from {}", weight_file)
            state_dict = torch.load(weight_file, map_location=_DEVICE)
            try:
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing:
                    logger.warning(f"Missing keys: {len(missing)} keys")
                if unexpected:
                    logger.warning(f"Unexpected keys: {len(unexpected)} keys")
                return model.eval()
            except Exception as e:
                logger.warning(f"Non-strict loading failed: {e!s}")
                model.load_state_dict(state_dict, strict=True)
                return model.eval()

    raise ModelNotFoundError(folder)
