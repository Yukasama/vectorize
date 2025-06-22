# ruff: noqa: S101

"""Tests for the training endpoint with valid data using real test datasets."""

import json
import re
import shutil
import uuid
from pathlib import Path

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from httpx import Response

from vectorize.config import settings

MINILM_MODEL_TAG = "models--sentence-transformers--all-MiniLM-L6-v2"
DATASET_ID_1 = "0b30b284-f7fe-4e6c-a270-17cafc5b5bcb"
DATASET_ID_2 = "0a9d5e87-e497-4737-9829-2070780d10df"
DEFAULT_EPOCHS = 3
DEFAULT_LR = 0.00005
DEFAULT_BATCH_SIZE = 8
TEST_DATA_DIR = Path("test_data/training/datasets")

HTTP_200_OK = status.HTTP_200_OK
HTTP_202_ACCEPTED = status.HTTP_202_ACCEPTED
HTTP_404_NOT_FOUND = status.HTTP_404_NOT_FOUND


def ensure_minilm_model_available() -> None:
    """Ensure the required model files are present for training tests."""
    src = Path("test_data/training/models--sentence-transformers--all-MiniLM-L6-v2")
    dst = settings.model_upload_dir / "models--sentence-transformers--all-MiniLM-L6-v2"
    if not dst.exists() and src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)


def ensure_test_datasets_exist() -> None:
    """Ensure the test datasets exist and are valid."""
    dataset_files = [
        TEST_DATA_DIR / f"__rm_-rf__2F_{DATASET_ID_1}.jsonl",
        TEST_DATA_DIR / f"__rm_-rf__2F_{DATASET_ID_2}.jsonl",
    ]

    for dataset_file in dataset_files:
        assert dataset_file.exists(), f"Test dataset not found: {dataset_file}"

        # Verify file is not empty and has valid JSON
        with dataset_file.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            assert len(lines) > 0, f"Dataset file is empty: {dataset_file}"


def extract_task_id_from_response(response: Response) -> str:
    """Extract task_id from training response (from Location header or JSON body).

    Args:
        response: FastAPI TestClient response

    Returns:
        task_id as string

    Raises:
        AssertionError: If no task_id can be found
    """
    task_id = None

    # Try to extract from Location header first
    if response.headers.get("Location"):
        match = re.search(
            r"/training/([a-f0-9\-]+)/status",
            response.headers["Location"],
        )
        if match:
            task_id = match.group(1)

    # Fallback to JSON body if available
    elif (
        response.content
        and response.headers.get("content-type", "").startswith("application/json")
    ):
        data = response.json()
        task_id = data.get("task_id")

    assert task_id, "No task_id found in response or headers"
    return task_id


@pytest.mark.training
class TestTrainingValid:
    """Tests for the training endpoint with valid data using real test datasets."""

    @staticmethod
    def test_valid_training(client: TestClient) -> None:
        """Test training with valid data and check response and status tracking."""
        ensure_minilm_model_available()
        ensure_test_datasets_exist()

        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "train_dataset_ids": [DATASET_ID_1],
            "val_dataset_id": DATASET_ID_2,
            "epochs": DEFAULT_EPOCHS,
            "learning_rate": DEFAULT_LR,
            "per_device_train_batch_size": DEFAULT_BATCH_SIZE,
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code == HTTP_202_ACCEPTED
        task_id = extract_task_id_from_response(response)
        status_response = client.get(f"/training/{task_id}/status")
        assert status_response.status_code == HTTP_200_OK
        status_data = status_response.json()
        assert status_data["status"] in {"Q", "R", "D", "F"}

        # Import an den Dateianfang verschoben (PLC0415)
        trained_models_dir = settings.model_upload_dir / "trained_models"
        model_dirs = trained_models_dir.glob("*-finetuned-*")
        for d in model_dirs:
            shutil.rmtree(d, ignore_errors=True)

    @staticmethod
    def test_get_training_status(client: TestClient) -> None:
        """Test the status endpoint for a training task with random ID (should fail)."""
        random_id = str(uuid.uuid4())
        response = client.get(f"/training/{random_id}/status")
        assert response.status_code == HTTP_404_NOT_FOUND
        data = response.json()
        assert "not found" in str(data).lower()

    @staticmethod
    def test_training_with_single_dataset(client: TestClient) -> None:
        """Test training with only one dataset (should use auto-split 90/10)."""
        ensure_minilm_model_available()
        ensure_test_datasets_exist()

        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "train_dataset_ids": [DATASET_ID_1],
            "epochs": 1,
            "learning_rate": DEFAULT_LR,
            "per_device_train_batch_size": DEFAULT_BATCH_SIZE,
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code == HTTP_202_ACCEPTED
        task_id = extract_task_id_from_response(response)

        # Verify that we can get the task status
        status_response = client.get(f"/training/{task_id}/status")
        assert status_response.status_code == HTTP_200_OK

    @staticmethod
    def test_training_with_multiple_datasets(client: TestClient) -> None:
        """Test training with multiple training datasets."""
        ensure_minilm_model_available()
        ensure_test_datasets_exist()

        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "train_dataset_ids": [DATASET_ID_1, DATASET_ID_2],
            "epochs": 1,
            "learning_rate": DEFAULT_LR,
            "per_device_train_batch_size": DEFAULT_BATCH_SIZE,
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code == HTTP_202_ACCEPTED
        task_id = extract_task_id_from_response(response)

        # Verify that we can get the task status
        status_response = client.get(f"/training/{task_id}/status")
        assert status_response.status_code == HTTP_200_OK

    @staticmethod
    def test_dataset_schema_validation(_client: TestClient) -> None:
        """Test that our test datasets have the correct schema."""
        ensure_test_datasets_exist()

        test_files = [
            TEST_DATA_DIR / f"__rm_-rf__2F_{DATASET_ID_1}.jsonl",
            TEST_DATA_DIR / f"__rm_-rf__2F_{DATASET_ID_2}.jsonl",
        ]

        for test_file in test_files:
            with test_file.open("r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            example = json.loads(line)
                        except json.JSONDecodeError:
                            pytest.fail(
                                f"Invalid JSON in {test_file.name} line {line_num}"
                            )

                        required_fields = ["question", "positive", "negative"]
                        for field in required_fields:
                            assert field in example, (
                                f"Missing '{field}' in {test_file.name} line {line_num}"
                            )
                            assert isinstance(
                                example[field], str
                            ), (
                                f"Field '{field}' should be string in {test_file.name} "
                                f"line {line_num}"
                            )
                            assert example[field].strip(), (
                                f"Field '{field}' should not be empty in "
                                f"{test_file.name} line {line_num}"
                            )
