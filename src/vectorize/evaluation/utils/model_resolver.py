"""Model path resolution utilities."""

from pathlib import Path

from vectorize.training.utils.safetensors_finder import find_safetensors_file

__all__ = ["resolve_model_path"]


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
    # Convert database model_tag to filesystem path
    # Handle different model tag formats:
    # 1. HuggingFace: "sentence-transformers_all-MiniLM-L6-v2" → "models--sentence-transformers--all-MiniLM-L6-v2"
    # 2. Trained: "trained_models/sentence-transformers_all-MiniLM-L6-v2-finetuned-..." → "trained_models/sentence-transformers_all-MiniLM-L6-v2-finetuned-..."
    
    if model_tag.startswith("trained_models/"):
        # Trained models: use as-is, they already have the correct path format
        filesystem_model_tag = model_tag
    else:
        # HuggingFace models: convert underscore format to dash format
        filesystem_model_tag = model_tag.replace("_", "--")
        if not filesystem_model_tag.startswith("models--"):
            filesystem_model_tag = f"models--{filesystem_model_tag}"
    
    base_path = Path("data/models") / filesystem_model_tag

    if not base_path.exists():
        raise FileNotFoundError(f"Model directory not found: {base_path}")

    snapshots_dir = base_path / "snapshots"
    if snapshots_dir.exists():
        snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
        if snapshot_dirs:
            candidate_path = snapshot_dirs[0]
            if (candidate_path / "config.json").exists():
                return str(candidate_path)
            safetensors_path = find_safetensors_file(str(candidate_path))
            if safetensors_path:
                return str(candidate_path)

        raise FileNotFoundError(f"No valid model found in snapshots: {snapshots_dir}")

    if (base_path / "config.json").exists():
        return str(base_path)

    safetensors_path = find_safetensors_file(str(base_path))
    if safetensors_path:
        return str(Path(safetensors_path).parent)

    raise FileNotFoundError(f"No valid model files found in: {base_path}")
