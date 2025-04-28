# ruff: noqa: S101

"""Test the model upload functionality."""

import os
from pathlib import Path

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from txt2vec.app import app
from txt2vec.config.config import prefix

client = TestClient(app)

# Direkter Pfad zum test_models-Ordner
MODELS_FOLDER = "test_models"

# Dateinamen Konstanten
TEST_FILE_NAME = "pytorch_model.bin"
INVALID_FORMAT_NAME = "nopytorch.safetensors"
EMPTY_FILE_NAME = "pytorch_empty.pt"
FILE_TOO_LARGE_NAME = "pytorch_model_big.bin"
VALID_SAFETENSORS_NAME = "model.safetensors"
EMPTY_ZIP_FOLDER = "zip_empty"
ZIP_WITH_MODELS_FOLDER = "zip_different_models"

BASE_URL = f"{prefix}/uploads/models"


@pytest.mark.parametrize(
    "file_name,mime_type",
    [
        (TEST_FILE_NAME, "application/octet-stream"),
        (VALID_SAFETENSORS_NAME, "application/octet-stream"),
    ],
)
def test_upload_valid_model_files(file_name: str, mime_type: str) -> None:
    """Test uploading valid model files."""
    base_dir = Path(__file__).parent.parent / MODELS_FOLDER
    test_file_path = base_dir / file_name

    file_content = Path(test_file_path).read_bytes()
    files = {"files": (os.path.basename(test_file_path), file_content, mime_type)}

    response = client.post(
        BASE_URL,
        params={"model_name": "test-model", "description": "Test model"},
        files=files,
    )

    assert response.status_code == status.HTTP_201_CREATED
    assert "Location" in response.headers


def test_upload_empty_model_file() -> None:
    """Test uploading an empty model file."""
    base_dir = Path(__file__).parent.parent / MODELS_FOLDER
    test_file_path = base_dir / EMPTY_FILE_NAME

    file_content = Path(test_file_path).read_bytes()
    files = {
        "files": (
            os.path.basename(test_file_path),
            file_content,
            "application/octet-stream",
        )
    }

    response = client.post(
        BASE_URL,
        params={"model_name": "empty-model"},
        files=files,
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["code"] == "EMPTY_MODEL"


def test_upload_large_model_file() -> None:
    """Test uploading a large model file."""
    base_dir = Path(__file__).parent.parent / MODELS_FOLDER
    test_file_path = base_dir / FILE_TOO_LARGE_NAME

    # Mock max_upload_size to be smaller for the test
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("txt2vec.upload.local_service.max_upload_size", 1024)  # 1KB limit

        file_content = Path(test_file_path).read_bytes()
        files = {
            "files": (
                os.path.basename(test_file_path),
                file_content,
                "application/octet-stream",
            )
        }

        response = client.post(
            BASE_URL,
            params={"model_name": "big-model"},
            files=files,
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert response.json()["code"] == "MODEL_TOO_LARGE"


def test_upload_invalid_pytorch_file() -> None:
    """Test uploading an invalid PyTorch file."""
    base_dir = Path(__file__).parent.parent / MODELS_FOLDER
    test_file_path = base_dir / EMPTY_FILE_NAME

    file_content = Path(test_file_path).read_bytes()
    files = {
        "files": (
            os.path.basename(test_file_path),
            file_content,
            "application/octet-stream",
        )
    }

    response = client.post(
        BASE_URL,
        params={"model_name": "invalid-model"},
        files=files,
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["code"] == "INVALID_MODEL"


def test_upload_non_pytorch_file() -> None:
    """Test uploading a file with unsupported extension."""
    base_dir = Path(__file__).parent.parent / MODELS_FOLDER
    test_file_path = base_dir / INVALID_FORMAT_NAME

    file_content = Path(test_file_path).read_bytes()
    files = {
        "files": (
            os.path.basename(test_file_path),
            file_content,
            "application/octet-stream",
        )
    }

    response = client.post(
        BASE_URL,
        params={"model_name": "wrong-format"},
        files=files,
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["code"] == "UNSUPPORTED_FORMAT"


def test_upload_empty_zip() -> None:
    """Test uploading an empty ZIP file."""
    base_dir = Path(__file__).parent.parent / MODELS_FOLDER / EMPTY_ZIP_FOLDER
    # Wir suchen nach ZIP-Dateien im Ordner
    zip_files = list(base_dir.glob("*.zip"))
    if not zip_files:
        pytest.skip("Keine ZIP-Dateien im Ordner gefunden.")

    test_file_path = zip_files[0]
    file_content = Path(test_file_path).read_bytes()
    files = {
        "files": (os.path.basename(test_file_path), file_content, "application/zip")
    }

    response = client.post(
        BASE_URL,
        params={"model_name": "empty-zip", "extract_zip": "true"},
        files=files,
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["code"] == "NO_VALID_MODELS"


def test_upload_zip_with_models() -> None:
    """Test uploading a ZIP with model files."""
    base_dir = Path(__file__).parent.parent / MODELS_FOLDER / ZIP_WITH_MODELS_FOLDER
    # Wir suchen nach ZIP-Dateien im Ordner
    zip_files = list(base_dir.glob("*.zip"))
    if not zip_files:
        pytest.skip("Keine ZIP-Dateien im Ordner gefunden.")

    test_file_path = zip_files[0]
    file_content = Path(test_file_path).read_bytes()
    files = {
        "files": (os.path.basename(test_file_path), file_content, "application/zip")
    }

    response = client.post(
        BASE_URL,
        params={"model_name": "zip-models", "extract_zip": "true"},
        files=files,
    )

    assert response.status_code == status.HTTP_201_CREATED
    assert "Location" in response.headers


def test_no_files_provided() -> None:
    """Test uploading with no files."""
    response = client.post(
        BASE_URL,
        params={"model_name": "no-files"},
        files={},
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["code"] == "INVALID_MODEL"
