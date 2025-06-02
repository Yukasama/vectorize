"""Extraction utilities for model ZIP files."""

import shutil
import tempfile
import zipfile
from pathlib import Path

from fastapi import UploadFile
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.ai_model import AIModel, ModelSource
from vectorize.ai_model.exceptions import ModelNotFoundError
from vectorize.ai_model.repository import get_ai_model_db, save_ai_model_db
from vectorize.config.config import settings

from ..exceptions import (
    ModelAlreadyExistsError,
    ModelTooLargeError,
    NoValidModelsFoundError,
)
from .zip_validator import validate_model_files

__all__ = ["process_model_directory", "process_single_model", "save_zip_to_temp"]


async def save_zip_to_temp(file: UploadFile) -> Path:
    """Save ZIP file to a temporary location.

    Args:
        file: The uploaded ZIP file

    Returns:
        Path to the saved temporary file

    Raises:
        ModelTooLargeError: If the file exceeds maximum upload size
    """
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    size = 0
    chunk_size = 1024 * 1024  # 1MB chunks
    with Path.open(temp_path, "wb") as dest_file:
        while chunk := await file.read(chunk_size):
            size += len(chunk)
            if size > settings.model_max_upload_size:
                Path.unlink(temp_path)
                raise ModelTooLargeError(size)
            dest_file.write(chunk)

    await file.seek(0)

    return temp_path


async def process_model_directory(
    zip_ref: zipfile.ZipFile,
    model_name: str,
    file_paths: list[str],
    base_dir: Path,
    db: AsyncSession,
) -> tuple[Path, str]:
    """Process a single model directory from the ZIP file.

    Args:
        zip_ref: Open ZIP file reference
        model_name: Name for this specific model
        file_paths: List of paths within this model's directory
        base_dir: Base directory where to extract files
        db: Database session for persistence

    Returns:
        Tuple of (model directory path, model database ID)

    Raises:
        NoValidModelsFoundError: When no valid model files were found
    """
    safe_model_name = "".join(c if c.isalnum() else "_" for c in model_name)
    try:
        await get_ai_model_db(db, safe_model_name)

        raise ModelAlreadyExistsError(
            f"Model with tag '{safe_model_name}' already exists"
        )
    except ModelNotFoundError:
        pass

    model_dir = base_dir / safe_model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    extracted_files = []

    common_prefix = None
    files = 2
    for path in file_paths:
        if path.endswith("/"):
            continue

        parts = path.split("/")
        if len(parts) >= files:
            prefix = "/".join(parts[:-1])
            if common_prefix is None or len(prefix) < len(common_prefix):
                common_prefix = prefix

    for file_path in file_paths:
        try:
            if file_path.endswith("/"):
                continue

            if common_prefix and file_path.startswith(common_prefix + "/"):
                relative_path = file_path[len(common_prefix) + 1 :]
            else:
                relative_path = Path(file_path).name

            target_path = model_dir / relative_path

            target_path.parent.mkdir(parents=True, exist_ok=True)

            with (
                zip_ref.open(file_path) as source,
                Path.open(target_path, "wb") as target,
            ):
                shutil.copyfileobj(source, target)

            extracted_files.append(target_path)
            logger.debug("Extracted {} to {}", file_path, target_path)
        except Exception as e:
            logger.error("Error extracting {}: {}", file_path, e)
            continue

    if not extracted_files:
        raise NoValidModelsFoundError(
            f"No files could be extracted for model {model_name}"
        )

    if not validate_model_files(extracted_files):
        raise NoValidModelsFoundError(
            f"No valid PyTorch model found in directory {model_name}"
        )

    ai_model = AIModel(
        model_tag=safe_model_name,
        name=model_name,
        source=ModelSource.LOCAL,
    )
    model_id = await save_ai_model_db(db, ai_model)

    return (model_dir, str(model_id))


async def process_single_model(
    zip_ref: zipfile.ZipFile,
    model_name: str,
    file_list: list,
    base_dir: Path,
    db: AsyncSession,
) -> tuple[Path, str]:
    """Process the ZIP contents as a single model.

    Args:
        zip_ref: Open ZIP file reference
        model_name: Name for the model
        file_list: List of all files in the ZIP
        base_dir: Base directory where to extract files
        db: Database session for persistence

    Returns:
        Tuple of (model directory path, model database ID)

    Raises:
        NoValidModelsFoundError: When no valid model files were found
    """
    safe_model_name = "".join(c if c.isalnum() else "_" for c in model_name)

    try:
        await get_ai_model_db(db, safe_model_name)

        raise ModelAlreadyExistsError(
            f"Model with tag '{safe_model_name}' already exists"
        )
    except ModelNotFoundError:
        pass

    model_dir = base_dir / safe_model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    extracted_files = []

    for file_info in file_list:
        if file_info.is_dir() or file_info.file_size <= 0:
            continue

        target_name = Path(file_info.filename).name
        target_path = model_dir / target_name

        with zip_ref.open(file_info) as source, Path.open(target_path, "wb") as target:
            shutil.copyfileobj(source, target)

        extracted_files.append(target_path)

    if not extracted_files:
        raise NoValidModelsFoundError("No valid files found in the archive")

    if not validate_model_files(extracted_files):
        raise NoValidModelsFoundError("No valid PyTorch model found in the archive")

    ai_model = AIModel(
        model_tag=safe_model_name,
        name=model_name,
        source=ModelSource.LOCAL,
    )
    model_id = await save_ai_model_db(db, ai_model)

    return (model_dir, str(model_id))
