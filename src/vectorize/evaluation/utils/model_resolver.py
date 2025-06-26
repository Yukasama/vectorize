"""Model path resolution utilities."""

from pathlib import Path

from vectorize.training.utils.safetensors_finder import find_safetensors_file

__all__ = ["resolve_model_path"]


def _normalize_model_tag(model_tag: str) -> str:
    """Normalize model tag to filesystem format."""
    if model_tag.startswith("trained_models/"):
        return model_tag

    filesystem_model_tag = model_tag.replace("_", "--")
    if not filesystem_model_tag.startswith("models--"):
        filesystem_model_tag = f"models--{filesystem_model_tag}"

    return filesystem_model_tag


def _find_valid_model_in_path(path: Path) -> str | None:
    """Find valid model files in the given path."""
    if (path / "config.json").exists():
        return str(path)

    safetensors_path = find_safetensors_file(str(path))
    if safetensors_path:
        return str(path)

    return None


def _resolve_snapshots_model(snapshots_dir: Path) -> str:
    """Resolve model path from snapshots directory."""
    snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
    if not snapshot_dirs:
        raise FileNotFoundError(f"No valid model found in snapshots: {snapshots_dir}")

    candidate_path = snapshot_dirs[0]
    model_path = _find_valid_model_in_path(candidate_path)
    if model_path:
        return model_path

    raise FileNotFoundError(f"No valid model found in snapshots: {snapshots_dir}")


def resolve_model_path(model_tag: str) -> str:
    """Resolve model tag to actual model path.

    Uses recursive search to find the correct model directory structure.
    Works with both HuggingFace cache format and direct model directories.

    Args:
        model_tag: Model tag/path

    Returns:
        str: Resolved model path containing model files

    Raises:
        FileNotFoundError: If model path cannot be resolved
    """
    filesystem_model_tag = _normalize_model_tag(model_tag)
    base_path = Path("data/models") / filesystem_model_tag

    if not base_path.exists():
        raise FileNotFoundError(f"Model directory not found: {base_path}")

    snapshots_dir = base_path / "snapshots"
    if snapshots_dir.exists():
        return _resolve_snapshots_model(snapshots_dir)

    model_path = _find_valid_model_in_path(base_path)
    if model_path:
        return model_path

    raise FileNotFoundError(f"No valid model files found in: {base_path}")
