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


def create_base_payload(**overrides: Any) -> dict[str, Any]:  # noqa: ANN401
    """Create a base training payload with optional overrides."""
    base = {
        "model_tag": MINILM_MODEL_TAG,
        "train_dataset_ids": [DATASET_ID_1],
        "epochs": 1,
        "learning_rate": 0.00005,
        "per_device_train_batch_size": 8,
    }
    return {**base, **overrides}


def ensure_minilm_model_available() -> None:
    """Ensure the required model files are present for training tests."""
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

    assert task_id, "No task_id found in response or headers"
    return task_id


def start_training_and_get_task_id(client: TestClient, payload: dict[str, Any]) -> str:
    """Start training and return task ID with standard assertions."""
    response = client.post("/training/train", json=payload)
    assert response.status_code == HTTP_202_ACCEPTED
    return extract_task_id_from_response(response)


def assert_valid_status_response(client: TestClient, task_id: str) -> dict[str, Any]:
    """Assert valid status response and return status data."""
    status_response = client.get(f"/training/{task_id}/status")
    assert status_response.status_code == HTTP_200_OK
    status_data = status_response.json()
    assert status_data["task_id"] == task_id
    assert status_data["status"] in {"Q", "R", "D", "F"}
    return status_data


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
    return response.json() if response.status_code == HTTP_200_OK else {}


def cleanup_trained_models() -> None:
    """Clean up any trained model directories."""
    trained_models_dir = settings.model_upload_dir / "trained_models"
    if trained_models_dir.exists():
        for model_dir in trained_models_dir.glob("*-finetuned-*"):
            shutil.rmtree(model_dir, ignore_errors=True)


def load_test_dataset(dataset_filename: str) -> list[dict[str, Any]]:
    """Load a test dataset from file and return as list of examples."""
    file_path = TEST_DATA_DIR / dataset_filename
    assert file_path.exists(), f"Test dataset not found: {file_path}"

    examples = []
    with file_path.open("r", encoding="utf-8") as f:
        examples.extend(json.loads(line) for line in f if line.strip())

    assert examples, f"No examples found in {file_path}"
    return examples


def validate_example_fields(example: dict[str, Any]) -> None:
    """Validate required fields in a dataset example."""
    required_fields = ["question", "positive", "negative"]
    for field in required_fields:
        assert field in example, f"Missing '{field}' field"
        assert isinstance(example[field], str), f"'{field}' should be string"
        assert example[field].strip(), f"'{field}' should not be empty"


def setup_training_test() -> None:
    """Common setup for training tests."""
    ensure_minilm_model_available()


def teardown_training_test() -> None:
    """Common teardown for training tests."""
    cleanup_trained_models()


def assert_training_response_structure(response: Response) -> str:
    """Assert training response has correct structure and return task_id."""
    assert response.status_code == HTTP_202_ACCEPTED
    assert "Location" in response.headers
    location = response.headers["Location"]
    assert "/training/" in location and "/status" in location

    task_id = extract_task_id_from_response(response)
    assert uuid.UUID(task_id)  # Should be valid UUID
    return task_id


def assert_status_response_fields(status_data: dict[str, Any], task_id: str) -> None:
    """Assert status response has all required fields."""
    required_fields = ["task_id", "status", "created_at"]
    for field in required_fields:
        assert field in status_data, f"Missing required field: {field}"

    assert status_data["task_id"] == task_id
    assert status_data["status"] in {"Q", "R", "D", "F"}
    assert isinstance(status_data["created_at"], str)


def run_training_test_with_payload(client: TestClient, payload: dict[str, Any]) -> str:
    """Run a complete training test cycle with setup, execution, and teardown."""
    setup_training_test()
    try:
        task_id = start_training_and_get_task_id(client, payload)
        assert_valid_status_response(client, task_id)
        return task_id
    finally:
        teardown_training_test()


def assert_error_response(response: Response, expected_status: int) -> None:
    """Assert response has expected error status."""
    assert response.status_code == expected_status


@pytest.mark.training
class TestTrainingComprehensive:
    """Comprehensive tests for training endpoints with real test data."""

    @staticmethod
    def test_training_single_dataset_with_split(client: TestClient) -> None:
        """Test training with single dataset (should auto-split 90/10)."""
        payload = create_base_payload()
        run_training_test_with_payload(client, payload)

    @staticmethod
    def test_training_multiple_datasets(client: TestClient) -> None:
        """Test training with multiple training datasets."""
        payload = create_base_payload(
            train_dataset_ids=[DATASET_ID_1, DATASET_ID_2],
            per_device_train_batch_size=4,
        )
        run_training_test_with_payload(client, payload)

    @staticmethod
    def test_training_parameter_validation(client: TestClient) -> None:
        """Test training parameter validation with various edge cases."""
        setup_training_test()

        # Test invalid parameters one by one
        invalid_params = [
            {"epochs": 0, "expected_msg": "epochs"},
            {"learning_rate": -0.1, "expected_msg": "learning_rate"},
            {"per_device_train_batch_size": 0, "expected_msg": "batch_size"},
        ]

        for invalid_param in invalid_params:
            payload = create_base_payload(**{
                k: v for k, v in invalid_param.items() if k != "expected_msg"
            })
            response = client.post("/training/train", json=payload)
            assert_error_response(response, HTTP_422_UNPROCESSABLE_ENTITY)

    @staticmethod
    def test_training_nonexistent_model(client: TestClient) -> None:
        """Test training with nonexistent model tag."""
        payload = create_base_payload(model_tag="nonexistent-model-tag")
        response = client.post("/training/train", json=payload)
        assert_error_response(response, HTTP_404_NOT_FOUND)

    @staticmethod
    def test_training_nonexistent_dataset(client: TestClient) -> None:
        """Test training with nonexistent dataset IDs."""
        setup_training_test()

        payload = create_base_payload(train_dataset_ids=[str(uuid.uuid4())])
        response = client.post("/training/train", json=payload)
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
        assert_error_response(response, HTTP_404_NOT_FOUND)

        # Test with invalid UUID format
        response = client.get("/training/invalid-uuid/status")
        assert response.status_code in {
            HTTP_400_BAD_REQUEST,
            HTTP_422_UNPROCESSABLE_ENTITY,
        }

    @staticmethod
    def test_training_response_structure(client: TestClient) -> None:
        """Test that training response has correct structure."""
        setup_training_test()
        try:
            payload = create_base_payload()
            response = client.post("/training/train", json=payload)
            assert_training_response_structure(response)
        finally:
            teardown_training_test()

    @staticmethod
    def test_training_status_response_structure(client: TestClient) -> None:
        """Test that status response has correct structure."""
        setup_training_test()
        try:
            payload = create_base_payload()
            task_id = start_training_and_get_task_id(client, payload)

            status_data = assert_valid_status_response(client, task_id)
            assert_status_response_fields(status_data, task_id)
        finally:
            teardown_training_test()

    @staticmethod
    def test_training_with_extreme_parameters(client: TestClient) -> None:
        """Test training with extreme but valid parameters."""
        payload = create_base_payload(
            epochs=1, learning_rate=1e-6, per_device_train_batch_size=1
        )
        run_training_test_with_payload(client, payload)

    @staticmethod
    def test_training_dataset_schema_validation() -> None:
        """Test that training works with our test data schema."""
        dataset_files = [
            f"__rm_-rf__2F_{DATASET_ID_2}.jsonl",
            f"__rm_-rf__2F_{DATASET_ID_1}.jsonl",
        ]

        for dataset_file in dataset_files:
            examples = load_test_dataset(dataset_file)
            assert len(examples) > 0, "Dataset should not be empty"

            # Check first few examples
            for example in examples[:3]:
                validate_example_fields(example)

    @staticmethod
    def test_training_concurrent_requests(client: TestClient) -> None:
        """Test handling of concurrent training requests."""
        setup_training_test()
        try:
            payload = create_base_payload()

            # Start two training tasks concurrently
            task_id1 = start_training_and_get_task_id(client, payload)
            task_id2 = start_training_and_get_task_id(client, payload)

            # Both should have different task IDs
            assert task_id1 != task_id2

            # Both should have valid status
            assert_valid_status_response(client, task_id1)
            assert_valid_status_response(client, task_id2)
        finally:
            teardown_training_test()

    @staticmethod
    def test_training_missing_required_fields(client: TestClient) -> None:
        """Test training request with missing required fields."""
        missing_field_tests = [
            {
                "field": "model_tag",
                "payload": {
                    "train_dataset_ids": [DATASET_ID_1],
                    "epochs": 1,
                    "learning_rate": 0.00005,
                    "per_device_train_batch_size": 8,
                },
            },
            {
                "field": "train_dataset_ids",
                "payload": {
                    "model_tag": MINILM_MODEL_TAG,
                    "epochs": 1,
                    "learning_rate": 0.00005,
                    "per_device_train_batch_size": 8,
                },
            },
            {
                "field": "train_dataset_ids (empty)",
                "payload": {
                    "model_tag": MINILM_MODEL_TAG,
                    "train_dataset_ids": [],
                    "epochs": 1,
                    "learning_rate": 0.00005,
                    "per_device_train_batch_size": 8,
                },
            },
        ]

        for test_case in missing_field_tests:
            response = client.post("/training/train", json=test_case["payload"])
            assert_error_response(response, HTTP_422_UNPROCESSABLE_ENTITY)
