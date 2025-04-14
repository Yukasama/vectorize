# ruff: noqa: S101

"""Test the dataset upload functionality."""

import os
from pathlib import Path

import pytest
from fastapi import status
from httpx import post

from txt2vec.config.config import BASE_URL

TRAINING_FOLDER = "testing_data"
CSV_TEST_FILE = "trainingdata.csv"
JSON_TEST_FILE = "trainingdata.json"
XML_TEST_FILE = "trainingdata.xml"


@pytest.mark.parametrize(
    "file_name,mime_type",
    [
        (CSV_TEST_FILE, "text/csv"),
        (JSON_TEST_FILE, "application/json"),
        (XML_TEST_FILE, "application/xml"),
    ],
)
def test_dataset_upload_multiple_formats(file_name: str, mime_type: str) -> None:
    """Parametrized test for uploading multiple file formats."""
    base_dir = Path(__file__).parent.parent / TRAINING_FOLDER
    test_file_path = base_dir / file_name

    file_content = Path(test_file_path).read_bytes()
    files = {"file": (os.path.basename(test_file_path), file_content, mime_type)}

    response = post(f"{BASE_URL}datasets", files=files)

    assert response.status_code == status.HTTP_201_CREATED
