"""Validation utilities for model ZIP files."""

import zipfile
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


def get_toplevel_directories(zip_file: zipfile.ZipFile) -> dict[str, list[str]]:
    """Identify model directories in the ZIP and group their contents.

    Args:
        zip_file: The opened ZIP file object

    Returns:
        Dictionary mapping potential model directory names to their contents
    """
    all_files = zip_file.namelist()
    potential_model_dirs = set()
    file_count = 2

    for file_path in all_files:
        parts = file_path.split("/")
        if len(parts) >= file_count and any(
            file_path.lower().endswith(ext)
            for ext in [".pt", ".pth", ".bin", ".model", ".safetensors"]
        ):
            potential_model_dirs.add("/".join(parts[:-1]))

    model_dirs = set()
    for dir1 in sorted(potential_model_dirs):
        if not any(
            dir1 != dir2 and dir1.startswith(f"{dir2}/")
            for dir2 in potential_model_dirs
        ):
            model_dirs.add(dir1)

    dir_files = {}
    for model_dir in model_dirs:
        dir_files[model_dir] = [f for f in all_files if f.startswith(f"{model_dir}/")]

    if not dir_files:
        for file_path in all_files:
            if "/" in file_path:
                dir_name = file_path.split("/")[0]
                if dir_name not in dir_files:
                    dir_files[dir_name] = []
                dir_files[dir_name].append(file_path)

    return dir_files
