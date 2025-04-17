# ruff: noqa: S101

"""Test the dataset upload functionality."""

import os
from pathlib import Path

import pytest
from fastapi import status
from httpx import post

from txt2vec.config.config import app_config

server_config = app_config.get("server", {})
port = server_config.get("port", 8000)
prefix = server_config.get("prefix", "v1")


BASE_URL = f"http://localhost:{port}/{prefix}"
TRAINING_FOLDER = "testing_data"
TEST_FILE_NAME = "trainingdata"


@pytest.mark.parametrize(
    "file_name,mime_type",
    [
        (f"{TEST_FILE_NAME}.csv", "text/csv"),
        (f"{TEST_FILE_NAME}.json", "application/json"),
        (f"{TEST_FILE_NAME}.xml", "application/xml"),
    ],
)
def test_dataset_upload_multiple_formats(file_name: str, mime_type: str) -> None:
    """Parametrized test for uploading multiple file formats."""
    base_dir = Path(__file__).parent.parent / TRAINING_FOLDER
    test_file_path = base_dir / file_name

    print(port, prefix, BASE_URL)

    file_content = Path(test_file_path).read_bytes()
    files = {"file": (os.path.basename(test_file_path), file_content, mime_type)}

    response = post(f"{BASE_URL}/datasets", files=files)

    assert response.status_code == status.HTTP_201_CREATED
