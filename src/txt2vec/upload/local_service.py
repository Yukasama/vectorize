"""Service for handling PyTorch model uploads including ZIP files."""

import os
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any, Final

import torch
from fastapi import UploadFile
from loguru import logger

from txt2vec.config.config import max_upload_size, model_upload_dir
from txt2vec.datasets.service import _sanitize_filename
from txt2vec.upload.exceptions import (
    EmptyModelError,
    InvalidModelError,
    InvalidZipError,
    ModelTooLargeError,
    NoValidModelsFoundError,
    UnsupportedModelFormatError,
)

# Gültige PyTorch-Modelldateiendungen
PYTORCH_EXTENSIONS: Final[set[str]] = {
    ".pt",
    ".pth",
    ".bin",
    ".model",
    ".weights",
    ".ckpt",
}


async def upload_embedding_model(
    files: list[UploadFile],
    model_name: str,
    extract_zip: bool = True,
) -> dict[str, Any]:
    """Verarbeitet PyTorch-Modell-Uploads mit ZIP-Unterstützung.

    Args:
        files: Liste der hochzuladenden Modelldateien
        model_name: Name für das Modell (wird als Verzeichnisname verwendet)
        extract_zip: Ob ZIP-Dateien extrahiert werden sollen (Standard: True)

    Returns:
        Dictionary mit Informationen über das hochgeladene Modell

    Raises:
        InvalidModelError: Wenn keine oder ungültige Dateien bereitgestellt werden
        EmptyModelError: Wenn eine Datei leer ist
        ModelTooLargeError: Wenn eine Datei zu groß ist
        UnsupportedModelFormatError: Wenn das Dateiformat nicht unterstützt wird
        InvalidZipError: Wenn eine ZIP-Datei ungültig ist
        NoValidModelsFoundError: Wenn keine gültigen PyTorch-Modelle gefunden wurden

    """
    if not files:
        raise InvalidModelError("No files provided for upload")

    # Sanitize model name using existing dataset function
    safe_model_name = _sanitize_filename(model_name)
    model_id = uuid.uuid4()
    model_dir = Path(model_upload_dir) / f"{safe_model_name}_{model_id}"
    model_dir.mkdir(parents=True, exist_ok=True)

    pytorch_file_count = 0

    try:
        pytorch_file_count = await _process_uploaded_files(
            files, model_dir, extract_zip
        )

        if pytorch_file_count == 0:
            raise NoValidModelsFoundError(
                "No valid PyTorch model files found in the upload"
            )

        return {
            "model_id": str(model_id),
            "model_name": safe_model_name,
            "model_dir": str(model_dir),
            "file_count": pytorch_file_count,
        }
    except (
        ModelTooLargeError,
        InvalidModelError,
        EmptyModelError,
        UnsupportedModelFormatError,
        InvalidZipError,
        NoValidModelsFoundError,
    ) as e:
        _cleanup_model_dir(model_dir)
        raise e
    except Exception as e:
        _cleanup_model_dir(model_dir)
        logger.exception("Unexpected error during model upload")
        raise InvalidModelError("Error processing upload") from e


async def _process_uploaded_files(
    files: list[UploadFile],
    model_dir: Path,
    extract_zip: bool,
) -> int:
    """Process each uploaded file and count valid PyTorch models."""
    pytorch_file_count = 0

    for file in files:
        if not file.filename:
            logger.warning("File without filename skipped")
            continue

        temp_path = None
        try:
            file_size, temp_path = await _process_single_file(file)
            if file_size <= 1:
                raise EmptyModelError(f"File '{file.filename}' is empty")

            filename = Path(file.filename).name
            file_ext = Path(filename).suffix.lower()

            if extract_zip and filename.lower().endswith(".zip"):
                pytorch_file_count += _handle_zip_file(temp_path, model_dir)
            elif file_ext in PYTORCH_EXTENSIONS:
                pytorch_file_count += _handle_pytorch_file(
                    temp_path, model_dir, filename
                )
            else:
                raise UnsupportedModelFormatError(
                    f"File '{file.filename}' has unsupported format. "
                    f"Supported: {', '.join(PYTORCH_EXTENSIONS)}"
                )
        except (ModelTooLargeError, EmptyModelError, UnsupportedModelFormatError) as e:
            # Den Fehler logger, aber explizit die original Exception weiterleiten
            logger.error(f"Error processing file {file.filename}: {e!s}")
            # Wichtig: Cleanup vor dem Raise
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    logger.warning(f"Failed to delete temporary file: {temp_path}")
            raise  # Explizites re-raise der original Exception
        finally:
            # Nur Cleanup der temporären Datei, wenn sie noch existiert und nicht durch eine Exception behandelt wurde
            if temp_path and os.path.exists(temp_path) and "e" not in locals():
                try:
                    os.unlink(temp_path)
                except Exception:
                    logger.warning(f"Failed to delete temporary file: {temp_path}")
            await file.seek(0)

    return pytorch_file_count


async def _process_single_file(file: UploadFile) -> tuple[int, str]:
    """Process a single uploaded file into a temporary file."""
    logger.debug(f"Starting to process file: {file.filename}")
    file_size = 0
    chunk_size = 1024 * 1024  # 1 MB chunks
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp_path = temp.name
            while chunk := await file.read(chunk_size):
                file_size += len(chunk)
                logger.debug(
                    f"Read chunk, total size now: {file_size}/{max_upload_size}"
                )
                if file_size > max_upload_size:
                    logger.warning(
                        f"File size ({file_size}) exceeds limit ({max_upload_size})"
                    )
                    raise ModelTooLargeError(file_size)
                temp.write(chunk)
            logger.debug(
                f"Completed processing file: {file.filename}, size: {file_size}"
            )
            return file_size, temp_path
    except ModelTooLargeError:
        # Bei Größenfehler die Temp-Datei löschen und den Fehler weiterleiten
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise  # Wichtig: den original ModelTooLargeError weiterleiten
    except Exception as e:
        # Bei anderen Fehlern die Temp-Datei löschen und einen allgemeinen Fehler auslösen
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        logger.exception(f"Error processing file {file.filename}")
        raise InvalidModelError(f"Error processing file {file.filename}: {e!s}") from e


def _handle_zip_file(zip_path: str, model_dir: Path) -> int:
    """Handle ZIP file extraction and return count of valid models."""
    if not zipfile.is_zipfile(zip_path):
        logger.warning(f"Invalid ZIP file: {zip_path}")
        raise InvalidZipError("File is not a valid ZIP file")

    extracted = _extract_pytorch_models(zip_path, model_dir)
    logger.debug(f"Extracted {extracted} PyTorch models from ZIP")
    return extracted


def _handle_pytorch_file(file_path: str, model_dir: Path, filename: str) -> int:
    """Handle PyTorch model file and return 1 if valid, otherwise raise."""
    if _is_valid_pytorch_model(file_path):
        dest_path = model_dir / filename
        shutil.move(file_path, dest_path)
        logger.debug(f"Saved valid PyTorch model: {filename}")
        return 1
    logger.warning(f"Invalid PyTorch model: {filename}")
    raise InvalidModelError("File is not a valid PyTorch model")


def _cleanup_model_dir(model_dir: Path) -> None:
    """Clean up the model directory if it exists."""
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)


def _is_valid_pytorch_model(file_path: str) -> bool:
    """Prüft, ob eine Datei ein gültiges PyTorch-Modell ist."""
    try:
        torch.load(file_path, map_location="cpu")
        return True
    except Exception as e:
        logger.debug(f"Invalid PyTorch model: {e!s}")
        return False


def _extract_pytorch_models(zip_path: str, extract_to: Path) -> int:
    """Extrahiert nur gültige PyTorch-Modelldateien aus einer ZIP-Datei."""
    count = 0

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.is_dir() or file_info.file_size == 0:
                continue

            file_ext = Path(file_info.filename).suffix.lower()
            if file_ext in PYTORCH_EXTENSIONS:
                temp_path = ""
                try:
                    with tempfile.NamedTemporaryFile(delete=False) as temp:
                        temp_path = temp.name

                    with (
                        zip_ref.open(file_info) as source,
                        open(temp_path, "wb") as target,
                    ):
                        shutil.copyfileobj(source, target)

                    if _is_valid_pytorch_model(temp_path):
                        target_path = extract_to / Path(file_info.filename).name
                        shutil.move(temp_path, target_path)
                        count += 1
                        logger.debug(f"Extracted valid model: {file_info.filename}")
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.unlink(temp_path)

    return count
