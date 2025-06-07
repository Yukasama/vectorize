"""Validation utilities for model ZIP files."""

import zipfile
from collections import defaultdict
from pathlib import Path

import torch
from loguru import logger

__all__ = ["get_toplevel_directories", "is_valid_zip", "validate_model_files"]


def validate_model_files(extracted_files: list[Path]) -> bool:
    """Check if at least one valid PyTorch model exists in the list of files.

    Args:
        extracted_files: List of paths to extracted files

    Returns:
        bool: True if at least one valid PyTorch model was found, False otherwise
    """
    valid_extensions = {".pt", ".pth", ".bin", ".model", ".safetensors"}

    for file_path in extracted_files:
        if file_path.suffix.lower() in valid_extensions:
            try:
                torch.load(file_path, map_location="cpu")  # NOSONAR
                return True
            except Exception:
                logger.debug("Invalid PyTorch model: {}", file_path)
                continue

    return False


def is_valid_zip(file_path: Path) -> bool:
    """Check if a file is a valid ZIP archive.

    Args:
        file_path: Path to the file to check

    Returns:
        bool: True if file is a valid ZIP archive, False otherwise
    """
    return zipfile.is_zipfile(file_path)


_MODEL_EXTS = (".pt", ".pth", ".bin", ".model", ".safetensors")


def get_toplevel_directories(zip_file: zipfile.ZipFile) -> dict[str, list[str]]:
    """Group ZIP entries by top-level model directory or first path segment.

    Args:
        zip_file: An opened ``zipfile.ZipFile`` instance.

    Returns:
        Dictionary mapping directory names to lists of file paths inside each
        directory.
    """
    all_files = zip_file.namelist()

    model_dirs = _collect_model_dirs(all_files)
    if model_dirs:
        return {
            md: [f for f in all_files if f.startswith(f"{md}/")] for md in model_dirs
        }

    grouped: dict[str, list[str]] = defaultdict(list)
    for path in all_files:
        if "/" in path:
            grouped[path.split("/", 1)[0]].append(path)
    return dict(grouped)


def _collect_model_dirs(all_files: list[str]) -> set[str]:
    """Return top-level directories that contain at least one model file.

    Args:
        all_files: List of paths obtained from ``zipfile.ZipFile.namelist``.

    Returns:
        Set of directory paths containing model files.
    """
    candidates = {
        "/".join(p.split("/")[:-1])
        for p in all_files
        if "/" in p and p.lower().endswith(_MODEL_EXTS)
    }

    return {
        d
        for d in candidates
        if not any(d != other and d.startswith(f"{other}/") for other in candidates)
    }
