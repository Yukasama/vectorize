"""Test module for model upload functionality using mocks."""

import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import status

BASE_URL = "/v1/uploads/models"
VALID_STATUS_CODES = {
    status.HTTP_400_BAD_REQUEST,
    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
}


def create_mock_file(filename: str, content: bytes = b"\x00" * 1024) -> tuple:
    """Create a mock for a file with customizable parameters."""
    mock_file = MagicMock()
    mock_file.filename = filename
    mock_file.read.return_value = content
    mock_file.seek.return_value = None

    file_dict = {"files": (filename, content, "application/octet-stream")}

    return mock_file, file_dict


class TestModelUploadWithMocks:
    """Test class for uploading models using mocks instead of real files."""

    @pytest.fixture(autouse=True)
    def setup(self) -> Generator[None, Any]:
        """Set up the test environment with required mocks."""
        with (
            patch("pathlib.Path.read_bytes", return_value=b"\x00" * 1024),
            patch("pathlib.Path.exists", return_value=True),
            patch("os.path.exists", return_value=True),
            patch("uuid.uuid4", return_value="test-uuid"),
            patch(
                "txt2vec.upload.local_service._is_valid_pytorch_model",
                return_value=True,
            ),
            patch("shutil.move") as self.mock_move,
            patch("tempfile.NamedTemporaryFile") as self.mock_temp,
        ):
            self.mock_temp_instance = MagicMock()
            self.mock_temp.return_value.__enter__.return_value = self.mock_temp_instance
            secure_temp_path = Path(tempfile.gettempdir()) / "test_temp_file"
            self.mock_temp_instance.name = str(secure_temp_path)

            yield

    @staticmethod
    def test_upload_valid_model_files(client: MagicMock) -> None:
        """Test uploading valid model files."""
        _, file_dict = create_mock_file("pytorch_model.bin")

        response = client.post(
            BASE_URL, params={"model_name": "test-model"}, files=file_dict
        )

        if response.status_code != status.HTTP_201_CREATED:
            pytest.fail(f"Expected status 201, got: {response.status_code}")
        if "Location" not in response.headers:
            pytest.fail("Header 'Location' not found in response")

    @staticmethod
    def test_upload_empty_model_file(client: MagicMock) -> None:
        """Test uploading an empty model file."""
        with patch(
            "txt2vec.upload.local_service._is_valid_pytorch_model",
            return_value=False,
        ):
            _, file_dict = create_mock_file("empty_model.pt", content=b"")

            response = client.post(
                BASE_URL, params={"model_name": "empty-model"}, files=file_dict
            )

            if response.status_code != status.HTTP_400_BAD_REQUEST:
                pytest.fail(f"Expected status 400, got: {response.status_code}")
            if response.json()["code"] != "EMPTY_FILE":
                pytest.fail(f"Expected code EMPTY_FILE, got: {response.json()['code']}")

    @staticmethod
    def test_upload_large_model_file(client: MagicMock) -> None:
        """Test uploading a model file that exceeds size limits."""
        with patch("txt2vec.upload.local_service.max_upload_size", 1024):
            _, file_dict = create_mock_file("large_model.bin", content=b"\x00" * 2048)

            response = client.post(
                BASE_URL, params={"model_name": "large-model"}, files=file_dict
            )

            if response.status_code != status.HTTP_400_BAD_REQUEST:
                pytest.fail(f"Expected status 400, got: {response.status_code}")

            expected_codes = {"MODEL_TOO_LARGE", "INVALID_FILE"}
            if response.json()["code"] not in expected_codes:
                pytest.fail(
                    f"Expected code in {expected_codes}, got: {response.json()['code']}"
                )

    @staticmethod
    def test_upload_invalid_pytorch_file(client: MagicMock) -> None:
        """Test uploading an invalid PyTorch file."""
        with patch(
            "txt2vec.upload.local_service._is_valid_pytorch_model",
            return_value=False,
        ):
            _, file_dict = create_mock_file("invalid_model.pt", content=b"not a model")

            response = client.post(
                BASE_URL, params={"model_name": "invalid-model"}, files=file_dict
            )

            if response.status_code != status.HTTP_400_BAD_REQUEST:
                pytest.fail(f"Expected status 400, got: {response.status_code}")
            if response.json()["code"] != "INVALID_FILE":
                pytest.fail(
                    f"Expected code INVALID_FILE, got: {response.json()['code']}"
                )

    @staticmethod
    def test_upload_non_pytorch_file(client: MagicMock) -> None:
        """Test uploading a file with unsupported extension."""
        _, file_dict = create_mock_file("model.unknown")

        response = client.post(
            BASE_URL, params={"model_name": "wrong-format"}, files=file_dict
        )

        if response.status_code not in VALID_STATUS_CODES:
            pytest.fail(
                f"Expected status in {VALID_STATUS_CODES}, got: {response.status_code}"
            )
        if response.json()["code"] != "UNSUPPORTED_FORMAT":
            pytest.fail(
                f"Expected code UNSUPPORTED_FORMAT, got: {response.json()['code']}"
            )

    @staticmethod
    def test_upload_empty_zip(client: MagicMock) -> None:
        """Test uploading an empty ZIP file."""
        with (
            patch("zipfile.is_zipfile", return_value=True),
            patch(
                "txt2vec.upload.local_service._extract_pytorch_models",
                return_value=0,
            ),
        ):
            _, file_dict = create_mock_file("empty.zip")

            response = client.post(
                BASE_URL,
                params={"model_name": "empty-zip", "extract_zip": "true"},
                files=file_dict,
            )

            if response.status_code != status.HTTP_400_BAD_REQUEST:
                pytest.fail(f"Expected status 400, got: {response.status_code}")

            expected_codes = {"INVALID_FILE", "NO_VALID_MODELS"}
            if response.json()["code"] not in expected_codes:
                pytest.fail(
                    f"Expected code in {expected_codes}, got: {response.json()['code']}"
                )

    @staticmethod
    def test_upload_zip_with_models(client: MagicMock) -> None:
        """Test uploading a ZIP file containing valid model files."""
        with (
            patch("zipfile.is_zipfile", return_value=True),
            patch(
                "txt2vec.upload.local_service._extract_pytorch_models",
                return_value=2,
            ),
        ):
            _, file_dict = create_mock_file("models.zip")

            response = client.post(
                BASE_URL,
                params={"model_name": "zip-models", "extract_zip": "true"},
                files=file_dict,
            )

            if response.status_code != status.HTTP_201_CREATED:
                pytest.fail(f"Expected status 201, got: {response.status_code}")
            if "Location" not in response.headers:
                pytest.fail("Header 'Location' not found in response")

    @staticmethod
    def test_no_files_provided(client: MagicMock) -> None:
        """Test handling the case when no files are provided."""
        response = client.post(
            BASE_URL,
            params={"model_name": "no-files"},
            files={},
        )

        if response.status_code != status.HTTP_400_BAD_REQUEST:
            pytest.fail(f"Expected status 400, got: {response.status_code}")
