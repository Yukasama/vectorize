# ruff: noqa: S101

"""Tests for ZIP model upload functionality."""

import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path
from uuid import UUID

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from txt2vec.config.config import settings

from .utils import get_test_zip_file


@pytest.mark.asyncio
@pytest.mark.upload
class TestZipModelUpload:
    """Tests for uploading models via ZIP files."""

    _base_dir = Path(__file__).parent.parent / "test_data" / "local_models"

    _valid_zip = _base_dir / "local_test_model.zip"

    _empty_zip = _base_dir / "empty_model.zip"

    _no_model_zip = _base_dir / "no_model.zip"

    _duplicate_model_zip = _base_dir / "duplicate_model.zip"

    _multiple_models_zip = _base_dir / "multiple_model.zip"

    @staticmethod
    @pytest.fixture
    async def temp_model_dir() -> AsyncGenerator[str]:
        """Create a temporary directory for model files during testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_property = type(settings).model_upload_dir

            temp_property = property(lambda _: Path(temp_dir))

            type(settings).model_upload_dir = temp_property

            try:
                yield temp_dir
            finally:
                type(settings).model_upload_dir = original_property

    @staticmethod
    async def test_valid_zip_upload(client: TestClient, temp_model_dir: str) -> None:
        """Test uploading a valid ZIP file with model files."""
        files = get_test_zip_file(TestZipModelUpload._valid_zip)

        response = client.post(
            "/uploads/local_models", params={"model_name": "test_model"}, files=files
        )

        assert response.status_code == status.HTTP_201_CREATED

        assert "Location" in response.headers

        model_id = response.headers["Location"].split("/")[-1]

        assert UUID(model_id)

        temp_files = list(Path(temp_model_dir).glob("**/*"))

        assert len(temp_files) > 0

    @staticmethod
    async def test_invalid_file_extension(client: TestClient) -> None:
        """Test uploading a file with an invalid extension."""
        invalid_file_path = TestZipModelUpload._base_dir / "invalid_file.txt"
        if not invalid_file_path.exists():
            with invalid_file_path.open("w") as f:
                f.write("This is not a ZIP file")

        files = get_test_zip_file(invalid_file_path)

        response = client.post(
            "/uploads/local_models", params={"model_name": "invalid_model"}, files=files
        )

        assert response.status_code != status.HTTP_201_CREATED

        assert response.json()["code"] == "INVALID_FILE"

    @staticmethod
    async def test_empty_zip_upload(client: TestClient) -> None:
        """Test uploading an empty ZIP file."""
        files = get_test_zip_file(TestZipModelUpload._empty_zip)

        response = client.post(
            "/uploads/local_models",
            params={"model_name": "empty_zip_model", "extract_zip": "false"},
            files=files,
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST

        assert response.json()["code"] == "INVALID_FILE"

    @staticmethod
    async def test_zip_without_models(client: TestClient) -> None:
        """Test uploading a ZIP file without valid model files."""
        files = get_test_zip_file(TestZipModelUpload._no_model_zip)

        response = client.post(
            "/uploads/local_models",
            params={"model_name": "no_model_zip", "extract_zip": "true"},
            files=files,
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST

        assert response.json()["code"] == "INVALID_FILE"

    @staticmethod
    async def test_duplicate_model(client: TestClient) -> None:
        """Test uploading a ZIP file with a copy of the model."""
        files = get_test_zip_file(TestZipModelUpload._duplicate_model_zip)

        response = client.post(
            "/uploads/local_models",
            params={"model_name": "duplicate_model", "extract_zip": "true"},
            files=files,
        )
        assert response.status_code == status.HTTP_409_CONFLICT

        assert response.json()["code"] == "MODEL_ALREADY_EXISTS"

    @staticmethod
    async def test_multiple_model(client: TestClient, temp_model_dir: str) -> None:
        """Test uploading a ZIP file with a copy of the model."""
        files = get_test_zip_file(TestZipModelUpload._multiple_models_zip)
        file_count = 2

        response = client.post(
            "/uploads/local_models",
            params={"model_name": "multiple_models", "extract_zip": "true"},
            files=files,
        )
        assert response.status_code == status.HTTP_201_CREATED

        assert response.json()["message"] == "Successfully uploaded 2 models"

        model_dirs = [d for d in Path(temp_model_dir).iterdir() if d.is_dir()]
        assert len(model_dirs) >= file_count
