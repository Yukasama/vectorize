# ruff: noqa: S101

"""Test the dataset upload functionality."""

import os
from pathlib import Path

from fastapi import status
from httpx import post

from txt2vec.config.config import BASE_URL

TRAINING_FOLDER = "testing_data"
CSV_TEST_FILE = "trainingdata.csv"


def test_upload_dataset() -> None:
    """Test uploading a dataset file and receiving a 201 status code.

    This test:
    1. Locates a CSV test file in the testing_data directory
    2. Uploads it to the /v1/datasets endpoint
    3. Verifies the response has status code 201 (Created)
    4. Checks that the response contains expected fields
    """
    base_dir = Path(__file__).parent.parent / TRAINING_FOLDER
    test_file_path = base_dir / CSV_TEST_FILE

    file_content = Path(test_file_path).read_bytes()
    files = {"file": (os.path.basename(test_file_path), file_content, "text/csv")}

    response = post(f"{BASE_URL}datasets", files=files)

    assert response.status_code == status.HTTP_201_CREATED, (
        f"Expected 201 Created, got {response.status_code}: {response.text}"
    )

    response_data = response.json()
    assert "filename" in response_data, "Response missing 'filename' field"
    assert "rows" in response_data, "Response missing 'rows' field"
    assert "columns" in response_data, "Response missing 'columns' field"
    assert "dataset_type" in response_data, "Response missing 'dataset_type' field"
