# ruff: noqa: S101

"""Test the dataset upload functionality."""

import os
from pathlib import Path

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from txt2vec.app import app
from txt2vec.config.config import prefix

client = TestClient(app)


TRAINING_FOLDER = "testing_data"
TEST_FILE_NAME = "trainingdata"
INVALID_FORMAT_NAME = "trainingsdata_wrong_format.csv"
EMPTY_FILE_NAME = "trainingsdata_empty.csv"


@pytest.mark.parametrize(
    "file_name,mime_type",
    [
        (f"{TEST_FILE_NAME}.csv", "text/csv"),
        (f"{TEST_FILE_NAME}.json", "application/json"),
        (f"{TEST_FILE_NAME}.xml", "application/xml"),
    ],
)
def test_dataset_formats_upload(file_name: str, mime_type: str) -> None:
    """Parametrized test for uploading multiple file formats."""
    base_dir = Path(__file__).parent.parent / TRAINING_FOLDER
    test_file_path = base_dir / file_name

    file_content = Path(test_file_path).read_bytes()
    files = {"file": (os.path.basename(test_file_path), file_content, mime_type)}

    response = client.post(f"{prefix}/datasets", files=files)

    assert response.status_code == status.HTTP_201_CREATED


def test_dataset_invalid_format() -> None:
    """Test uploading an invalid file format."""
    base_dir = Path(__file__).parent.parent / TRAINING_FOLDER
    test_file_path = base_dir / INVALID_FORMAT_NAME

    file_content = Path(test_file_path).read_bytes()
    files = {"file": (os.path.basename(test_file_path), file_content, "text/csv")}

    response = client.post(f"{prefix}/datasets", files=files)

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["code"] == "INVALID_CSV_FORMAT"


def test_dataset_empty() -> None:
    """Test uploading an empty file."""
    base_dir = Path(__file__).parent.parent / TRAINING_FOLDER
    test_file_path = base_dir / EMPTY_FILE_NAME

    file_content = Path(test_file_path).read_bytes()
    files = {"file": (os.path.basename(test_file_path), file_content, "text/csv")}

    response = client.post(f"{prefix}/datasets", files=files)

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["code"] == "EMPTY_FILE"
