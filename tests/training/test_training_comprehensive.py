# ruff: noqa: S101

"""Comprehensive tests for training endpoints using real test data."""

import json
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from httpx import Response

from vectorize.config import settings

# Test constants
MINILM_MODEL_TAG = "models--sentence-transformers--all-MiniLM-L6-v2"
DATASET_ID_1 = "0b30b284-f7fe-4e6c-a270-17cafc5b5bcb"
DATASET_ID_2 = "0a9d5e87-e497-4737-9829-2070780d10df"
TEST_DATA_DIR = Path("test_data/training/datasets")

# HTTP Status codes
HTTP_200_OK = status.HTTP_200_OK
HTTP_202_ACCEPTED = status.HTTP_202_ACCEPTED
HTTP_400_BAD_REQUEST = status.HTTP_400_BAD_REQUEST
HTTP_404_NOT_FOUND = status.HTTP_404_NOT_FOUND
HTTP_422_UNPROCESSABLE_ENTITY = status.HTTP_422_UNPROCESSABLE_ENTITY

UUID_LENGTH = 36
UUID_HYPHEN_COUNT = 4


def ensure_minilm_model_available() -> None:
    """Ensure the required model files are present for training tests."""
    # Import an den Dateianfang verschoben (PLC0415)
    src = Path("test_data/training/models--sentence-transformers--all-MiniLM-L6-v2")
    dst = settings.model_upload_dir / "models--sentence-transformers--all-MiniLM-L6-v2"
    if not dst.exists() and src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)


def extract_task_id_from_response(response: Response) -> str:
    """Extract task_id from training response."""
    task_id = None

    if response.headers.get("Location"):
        match = re.search(
            r"/training/([a-f0-9\-]+)/status",
            response.headers["Location"],
        )
        if match:
            task_id = match.group(1)

    elif response.content and response.headers.get("content-type", "").startswith(
        "application/json"
    ):
        data = response.json()
        task_id = data.get("task_id")

    assert task_id, (
        "No task_id found in response or headers"
    )
    return task_id


def wait_for_task_completion(
    client: TestClient, task_id: str, max_wait: int = 30
) -> dict[str, Any]:
    """Wait for a training task to complete and return final status."""
    for _ in range(max_wait):
        response = client.get(f"/training/{task_id}/status")
        if response.status_code == HTTP_200_OK:
            data = response.json()
            if data["status"] in {"D", "F"}:
                return data
        time.sleep(1)

    # Return last known status
    response = client.get(f"/training/{task_id}/status")
    return (
        response.json() if response.status_code == HTTP_200_OK else {}
    )


def cleanup_trained_models() -> None:
    """Clean up any trained model directories."""
    trained_models_dir = settings.model_upload_dir / "trained_models"
    if trained_models_dir.exists():
        for model_dir in trained_models_dir.glob("*-finetuned-*"):
            shutil.rmtree(model_dir, ignore_errors=True)


def load_test_dataset(dataset_filename: str) -> list[dict[str, Any]]:
    """Load a test dataset from file and return as list of examples."""
    file_path = TEST_DATA_DIR / dataset_filename
    assert file_path.exists(), (
        f"Test dataset not found: {file_path}"
    )

    examples = []
    with file_path.open("r", encoding="utf-8") as f:
        examples.extend(json.loads(line) for line in f if line.strip())

    assert examples, (
        f"No examples found in {file_path}"
    )
    return examples


@pytest.mark.training
class TestTrainingComprehensive:
    """Comprehensive tests for training endpoints with real test data."""

    @staticmethod
    def test_training_single_dataset_with_split(client: TestClient) -> None:
        """Test training with single dataset (should auto-split 90/10)."""
        ensure_minilm_model_available()

        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "train_dataset_ids": [DATASET_ID_1],
            "epochs": 1,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8,
        }

        response = client.post("/training/train", json=payload)
        assert response.status_code == HTTP_202_ACCEPTED

        task_id = extract_task_id_from_response(response)
        assert uuid.UUID(task_id)  # Validate UUID format

        # Check initial status - allow for DONE since training might complete quickly
        status_response = client.get(f"/training/{task_id}/status")
        assert status_response.status_code == HTTP_200_OK
        status_data = status_response.json()
        assert status_data["status"] in {"Q", "R", "D", "F"}
        assert status_data["task_id"] == task_id

        cleanup_trained_models()

    @staticmethod
    def test_training_multiple_datasets(client: TestClient) -> None:
        """Test training with multiple training datasets."""
        ensure_minilm_model_available()

        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "train_dataset_ids": [DATASET_ID_1, DATASET_ID_2],
            "epochs": 1,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 4,
        }

        response = client.post("/training/train", json=payload)
        assert response.status_code == HTTP_202_ACCEPTED

        task_id = extract_task_id_from_response(response)

        # Check that training started successfully
        status_response = client.get(f"/training/{task_id}/status")
        assert status_response.status_code == HTTP_200_OK

        cleanup_trained_models()

    @staticmethod
    def test_training_parameter_validation(client: TestClient) -> None:
        """Test training parameter validation with various edge cases."""
        ensure_minilm_model_available()

        # Test with invalid epochs
        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "train_dataset_ids": [DATASET_ID_1],
            "epochs": 0,  # Invalid
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8,
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY

        # Test with invalid learning rate
        payload["epochs"] = 1
        payload["learning_rate"] = -0.1  # Invalid
        response = client.post("/training/train", json=payload)
        assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY

        # Test with invalid batch size
        payload["learning_rate"] = 0.00005
        payload["per_device_train_batch_size"] = 0  # Invalid
        response = client.post("/training/train", json=payload)
        assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY

    @staticmethod
    def test_training_nonexistent_model(client: TestClient) -> None:
        """Test training with nonexistent model tag."""
        payload = {
            "model_tag": "nonexistent-model-tag",
            "train_dataset_ids": [DATASET_ID_1],
            "epochs": 1,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8,
        }

        response = client.post("/training/train", json=payload)
        # Should return 404 for nonexistent model
        assert response.status_code == HTTP_404_NOT_FOUND

    @staticmethod
    def test_training_nonexistent_dataset(client: TestClient) -> None:
        """Test training with nonexistent dataset IDs."""
        ensure_minilm_model_available()

        fake_dataset_id = str(uuid.uuid4())
        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "train_dataset_ids": [fake_dataset_id],
            "epochs": 1,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8,
        }

        response = client.post("/training/train", json=payload)
        # Should either fail immediately or start and fail during execution
        assert response.status_code in {
            HTTP_400_BAD_REQUEST,
            HTTP_404_NOT_FOUND,
            HTTP_202_ACCEPTED,
        }

    @staticmethod
    def test_training_status_invalid_task_id(client: TestClient) -> None:
        """Test status endpoint with invalid task IDs."""
        # Test with completely random UUID
        random_id = str(uuid.uuid4())
        response = client.get(f"/training/{random_id}/status")
        assert response.status_code == HTTP_404_NOT_FOUND

        # Test with invalid UUID format
        response = client.get("/training/invalid-uuid/status")
        assert response.status_code in {
            HTTP_400_BAD_REQUEST,
            HTTP_422_UNPROCESSABLE_ENTITY,
        }

    @staticmethod
    def test_training_response_structure(client: TestClient) -> None:
        """Test that training response has correct structure."""
        ensure_minilm_model_available()

        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "train_dataset_ids": [DATASET_ID_1],
            "epochs": 1,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8,
        }

        response = client.post("/training/train", json=payload)
        assert response.status_code == HTTP_202_ACCEPTED

        # Check Location header
        assert "Location" in response.headers
        location = response.headers["Location"]
        assert "/training/" in location
        assert "/status" in location

        # Extract and validate task ID from location
        task_id = extract_task_id_from_response(response)
        assert uuid.UUID(task_id)  # Should be valid UUID

        cleanup_trained_models()

    @staticmethod
    def test_training_status_response_structure(client: TestClient) -> None:
        """Test that status response has correct structure."""
        ensure_minilm_model_available()

        # Start a training task
        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "train_dataset_ids": [DATASET_ID_1],
            "epochs": 1,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8,
        }

        response = client.post("/training/train", json=payload)
        assert response.status_code == HTTP_202_ACCEPTED
        task_id = extract_task_id_from_response(response)

        # Check status response structure
        status_response = client.get(f"/training/{task_id}/status")
        assert status_response.status_code == HTTP_200_OK

        status_data = status_response.json()
        required_fields = ["task_id", "status", "created_at"]
        for field in required_fields:
            assert field in status_data, (
                f"Missing required field: {field}"
            )

        assert status_data["task_id"] == task_id
        assert status_data["status"] in {"Q", "R", "D", "F"}

        # created_at should be ISO format timestamp
        assert isinstance(
            status_data["created_at"], str
        )

        cleanup_trained_models()

    @staticmethod
    def test_training_with_extreme_parameters(client: TestClient) -> None:
        """Test training with extreme but valid parameters."""
        ensure_minilm_model_available()

        # Test with minimum valid values
        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "train_dataset_ids": [DATASET_ID_1],
            "epochs": 1,  # Minimum
            "learning_rate": 1e-6,  # Very small
            "per_device_train_batch_size": 1,  # Minimum
        }

        response = client.post("/training/train", json=payload)
        assert response.status_code == HTTP_202_ACCEPTED

        task_id = extract_task_id_from_response(response)
        status_response = client.get(f"/training/{task_id}/status")
        assert status_response.status_code == HTTP_200_OK

        cleanup_trained_models()

    @staticmethod
    def test_training_dataset_schema_validation() -> None:
        """Test that training works with our test data schema."""
        # Verify our test datasets have the correct schema
        dataset1_examples = load_test_dataset(f"__rm_-rf__2F_{DATASET_ID_2}.jsonl")
        dataset2_examples = load_test_dataset(f"__rm_-rf__2F_{DATASET_ID_1}.jsonl")

        # Check that examples have required fields
        for examples in [dataset1_examples, dataset2_examples]:
            assert len(examples) > 0, "Dataset should not be empty"

            for example in examples[:3]:  # Check first few examples
                assert "question" in example, "Missing 'question' field"
                assert "positive" in example, "Missing 'positive' field"
                assert "negative" in example, "Missing 'negative' field"

                # Validate field types
                assert isinstance(
                    example["question"], str
                ), "'question' should be string"
                assert isinstance(
                    example["positive"], str
                ), "'positive' should be string"
                assert isinstance(
                    example["negative"], str
                ), "'negative' should be string"

                # Validate content is not empty
                assert example["question"].strip(), (
                    "'question' should not be empty"
                )
                assert example["positive"].strip(), (
                    "'positive' should not be empty"
                )
                assert example["negative"].strip(), (
                    "'negative' should not be empty"
                )

    @staticmethod
    def test_training_concurrent_requests(client: TestClient) -> None:
        """Test handling of concurrent training requests."""
        ensure_minilm_model_available()

        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "train_dataset_ids": [DATASET_ID_1],
            "epochs": 1,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8,
        }

        # Start first training
        response1 = client.post("/training/train", json=payload)
        assert response1.status_code == HTTP_202_ACCEPTED
        task_id1 = extract_task_id_from_response(response1)

        # Start second training immediately
        response2 = client.post("/training/train", json=payload)
        assert response2.status_code == HTTP_202_ACCEPTED
        task_id2 = extract_task_id_from_response(response2)

        # Both should have different task IDs
        assert task_id1 != task_id2

        # Both should have valid status
        status1 = client.get(f"/training/{task_id1}/status")
        status2 = client.get(f"/training/{task_id2}/status")
        assert status1.status_code == HTTP_200_OK
        assert status2.status_code == HTTP_200_OK

        cleanup_trained_models()

    @staticmethod
    def test_training_missing_required_fields(client: TestClient) -> None:
        """Test training request with missing required fields."""
        # Missing model_tag
        payload = {
            "train_dataset_ids": [DATASET_ID_1],
            "epochs": 1,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8,
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY

        # Missing train_dataset_ids
        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "epochs": 1,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8,
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY

        # Empty train_dataset_ids
        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "train_dataset_ids": [],
            "epochs": 1,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8,
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY
