# ruff: noqa: S101

"""Comprehensive tests for evaluation endpoints using real test data."""

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


def ensure_minilm_model_available() -> None:
    """Ensure the required model files are present for evaluation tests."""
    src = Path("test_data/training/models--sentence-transformers--all-MiniLM-L6-v2")
    dst = settings.model_upload_dir / "models--sentence-transformers--all-MiniLM-L6-v2"
    if not dst.exists() and src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)


def extract_task_id_from_response(response: Response) -> str:
    """Extract task_id from evaluation response."""
    task_id = None

    # Try Location header first
    if response.headers.get("Location"):
        match = re.search(
            r"/evaluation/([a-f0-9\-]+)/status",
            response.headers["Location"],
        )
        if match:
            task_id = match.group(1)

    # Fallback to JSON body
    elif response.content and response.headers.get("content-type", "").startswith(
        "application/json"
    ):
        data = response.json()
        task_id = data.get("task_id")

    assert task_id, "No task_id found in response or headers"
    return task_id


def wait_for_evaluation_completion(
    client: TestClient, task_id: str, max_wait: int = 60
) -> dict[str, Any]:
    """Wait for an evaluation task to complete and return final status."""
    for _ in range(max_wait):
        response = client.get(f"/evaluation/{task_id}/status")
        if response.status_code == HTTP_200_OK:
            data = response.json()
            if data["status"] in {"D", "F"}:
                return data
        time.sleep(1)

    # Return last known status
    response = client.get(f"/evaluation/{task_id}/status")
    return response.json() if response.status_code == HTTP_200_OK else {}


def load_test_dataset(dataset_filename: str) -> list[dict[str, Any]]:
    """Load a test dataset from file and return as list of examples."""
    file_path = TEST_DATA_DIR / dataset_filename
    assert file_path.exists(), f"Test dataset not found: {file_path}"

    examples = []
    with file_path.open("r", encoding="utf-8") as f:
        examples.extend(json.loads(line) for line in f if line.strip())

    assert examples, f"No examples found in {file_path}"
    return examples


def create_base_evaluation_payload(**overrides: Any) -> dict[str, Any]:  # noqa: ANN401
    """Create a base evaluation payload with optional overrides."""
    base = {
        "model_tag": MINILM_MODEL_TAG,
        "dataset_id": DATASET_ID_1,
    }
    return {**base, **overrides}


def setup_evaluation_test() -> None:
    """Common setup for evaluation tests."""
    ensure_minilm_model_available()


def start_evaluation_and_get_task_id(
    client: TestClient, payload: dict[str, Any]
) -> str:
    """Start evaluation and return task ID with standard assertions."""
    response = client.post("/evaluation/evaluate", json=payload)
    assert response.status_code == HTTP_202_ACCEPTED
    return extract_task_id_from_response(response)


def assert_valid_evaluation_status_response(
    client: TestClient, task_id: str
) -> dict[str, Any]:
    """Assert valid status response and return status data."""
    status_response = client.get(f"/evaluation/{task_id}/status")
    assert status_response.status_code == HTTP_200_OK
    status_data = status_response.json()
    assert status_data["task_id"] == task_id
    assert status_data["status"] in {"Q", "R", "D", "F"}
    return status_data


def assert_evaluation_response_structure(response: Response) -> str:
    """Assert evaluation response has correct structure and return task_id."""
    assert response.status_code == HTTP_202_ACCEPTED
    assert "Location" in response.headers
    location = response.headers["Location"]
    assert "/evaluation/" in location and "/status" in location

    task_id = extract_task_id_from_response(response)
    assert uuid.UUID(task_id)  # Should be valid UUID
    return task_id


def assert_evaluation_status_response_fields(
    status_data: dict[str, Any], task_id: str
) -> None:
    """Assert status response has all required fields."""
    required_fields = ["task_id", "status", "created_at"]
    for field in required_fields:
        assert field in status_data, f"Missing required field: {field}"

    assert status_data["task_id"] == task_id
    assert status_data["status"] in {"Q", "R", "D", "F"}
    assert isinstance(status_data["created_at"], str)


def assert_error_response(response: Response, expected_status: int) -> None:
    """Assert response has expected error status."""
    assert response.status_code == expected_status


def validate_dataset_example_fields(example: dict[str, Any]) -> None:
    """Validate required fields in a dataset example."""
    required_fields = ["question", "positive", "negative"]
    for field in required_fields:
        assert field in example, f"Missing '{field}' field"
        assert isinstance(example[field], str), f"'{field}' should be string"
        assert example[field].strip(), f"'{field}' should not be empty"


def run_evaluation_test_with_payload(
    client: TestClient, payload: dict[str, Any]
) -> str:
    """Run a complete evaluation test cycle with setup and execution."""
    setup_evaluation_test()
    task_id = start_evaluation_and_get_task_id(client, payload)
    assert_valid_evaluation_status_response(client, task_id)
    return task_id


@pytest.mark.evaluation
class TestEvaluationComprehensive:
    """Comprehensive tests for evaluation endpoints with real test data."""

    @staticmethod
    def test_evaluation_different_datasets(client: TestClient) -> None:
        """Test evaluation with different datasets."""
        setup_evaluation_test()

        # Test with first dataset
        payload1 = create_base_evaluation_payload()
        task_id1 = start_evaluation_and_get_task_id(client, payload1)

        # Test with second dataset
        payload2 = create_base_evaluation_payload(dataset_id=DATASET_ID_2)
        task_id2 = start_evaluation_and_get_task_id(client, payload2)

        # Both should have different task IDs
        assert task_id1 != task_id2

        # Both should have valid status
        assert_valid_evaluation_status_response(client, task_id1)
        assert_valid_evaluation_status_response(client, task_id2)

    @staticmethod
    def test_evaluation_nonexistent_model(client: TestClient) -> None:
        """Test evaluation with nonexistent model tag."""
        payload = create_base_evaluation_payload(model_tag="nonexistent-model-tag")
        response = client.post("/evaluation/evaluate", json=payload)
        # Should either fail immediately or start and fail during execution
        assert response.status_code in {HTTP_400_BAD_REQUEST, HTTP_202_ACCEPTED}

    @staticmethod
    def test_evaluation_nonexistent_dataset(client: TestClient) -> None:
        """Test evaluation with nonexistent dataset ID."""
        setup_evaluation_test()

        payload = create_base_evaluation_payload(dataset_id=str(uuid.uuid4()))
        response = client.post("/evaluation/evaluate", json=payload)
        # Should either fail immediately or start and fail during execution
        assert response.status_code in {
            HTTP_400_BAD_REQUEST,
            HTTP_404_NOT_FOUND,
            HTTP_202_ACCEPTED,
        }

    @staticmethod
    def test_evaluation_status_invalid_task_id(client: TestClient) -> None:
        """Test status endpoint with invalid task IDs."""
        # Test with completely random UUID
        random_id = str(uuid.uuid4())
        response = client.get(f"/evaluation/{random_id}/status")
        assert_error_response(response, HTTP_404_NOT_FOUND)

        # Test with invalid UUID format
        response = client.get("/evaluation/invalid-uuid/status")
        assert response.status_code in {
            HTTP_400_BAD_REQUEST,
            HTTP_422_UNPROCESSABLE_ENTITY,
        }

    @staticmethod
    def test_evaluation_response_structure(client: TestClient) -> None:
        """Test that evaluation response has correct structure."""
        setup_evaluation_test()

        payload = create_base_evaluation_payload()
        response = client.post("/evaluation/evaluate", json=payload)
        assert_evaluation_response_structure(response)

    @staticmethod
    def test_evaluation_status_response_structure(client: TestClient) -> None:
        """Test that status response has correct structure."""
        setup_evaluation_test()

        payload = create_base_evaluation_payload()
        task_id = start_evaluation_and_get_task_id(client, payload)

        status_data = assert_valid_evaluation_status_response(client, task_id)
        assert_evaluation_status_response_fields(status_data, task_id)

    @staticmethod
    def test_evaluation_results_endpoint(client: TestClient) -> None:
        """Test evaluation results endpoint."""
        setup_evaluation_test()

        payload = create_base_evaluation_payload()
        task_id = start_evaluation_and_get_task_id(client, payload)

        # Try to get results (may not be ready yet)
        results_response = client.get(f"/evaluation/{task_id}/results")
        # Should either be 404 (not ready) or 200 (ready)
        assert results_response.status_code in {HTTP_200_OK, HTTP_404_NOT_FOUND}

        if results_response.status_code == HTTP_200_OK:
            results_data = results_response.json()
            # Check that results have expected structure
            assert isinstance(results_data, dict)
            # Results structure may vary, but should be JSON serializable

    @staticmethod
    def test_evaluation_dataset_schema_validation() -> None:
        """Test that evaluation works with our test data schema."""
        # Verify our test datasets have the correct schema
        dataset_files = [
            f"__rm_-rf__2F_{DATASET_ID_2}.jsonl",
            f"__rm_-rf__2F_{DATASET_ID_1}.jsonl"
        ]

        for dataset_file in dataset_files:
            examples = load_test_dataset(dataset_file)
            assert len(examples) > 0, "Dataset should not be empty"

            # Check first few examples
            for example in examples[:3]:
                validate_dataset_example_fields(example)

    @staticmethod
    def test_evaluation_concurrent_requests(client: TestClient) -> None:
        """Test handling of concurrent evaluation requests."""
        setup_evaluation_test()

        # Start first evaluation
        payload1 = create_base_evaluation_payload()
        task_id1 = start_evaluation_and_get_task_id(client, payload1)

        # Start second evaluation immediately
        payload2 = create_base_evaluation_payload(dataset_id=DATASET_ID_2)
        task_id2 = start_evaluation_and_get_task_id(client, payload2)

        # Both should have different task IDs
        assert task_id1 != task_id2

        # Both should have valid status
        assert_valid_evaluation_status_response(client, task_id1)
        assert_valid_evaluation_status_response(client, task_id2)

    @staticmethod
    def test_evaluation_missing_required_fields(client: TestClient) -> None:
        """Test evaluation request with missing required fields."""
        # Missing model_tag
        payload = {"dataset_id": DATASET_ID_1}
        response = client.post("/evaluation/evaluate", json=payload)
        assert_error_response(response, HTTP_422_UNPROCESSABLE_ENTITY)

        # Missing dataset_id - backend accepts this and fails in background task
        payload = {"model_tag": MINILM_MODEL_TAG}
        response = client.post("/evaluation/evaluate", json=payload)
        assert_error_response(response, HTTP_202_ACCEPTED)

    @staticmethod
    def test_evaluation_invalid_request_data(client: TestClient) -> None:
        """Test evaluation with invalid request data types."""
        # Invalid model_tag type
        payload = {
            "model_tag": 123,  # Should be string
            "dataset_id": DATASET_ID_1,
        }
        response = client.post("/evaluation/evaluate", json=payload)
        assert_error_response(response, HTTP_422_UNPROCESSABLE_ENTITY)

        # Invalid dataset_id type
        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "dataset_id": 123,  # Should be string
        }
        response = client.post("/evaluation/evaluate", json=payload)
        assert_error_response(response, HTTP_422_UNPROCESSABLE_ENTITY)

    @staticmethod
    def test_evaluation_empty_string_parameters(client: TestClient) -> None:
        """Test evaluation with empty string parameters."""
        # Empty model_tag - backend accepts this and fails in background task
        payload = create_base_evaluation_payload(model_tag="")
        response = client.post("/evaluation/evaluate", json=payload)
        assert_error_response(response, HTTP_202_ACCEPTED)

        # Empty dataset_id - backend also accepts this and fails in background task
        payload = create_base_evaluation_payload(dataset_id="")
        response = client.post("/evaluation/evaluate", json=payload)
        assert_error_response(response, HTTP_202_ACCEPTED)

    @staticmethod
    def test_evaluation_status_field_validation(client: TestClient) -> None:
        """Test that evaluation status contains all expected fields."""
        setup_evaluation_test()

        payload = create_base_evaluation_payload()
        task_id = start_evaluation_and_get_task_id(client, payload)

        status_data = assert_valid_evaluation_status_response(client, task_id)

        # Field type validation
        assert isinstance(status_data["task_id"], str)
        assert isinstance(status_data["status"], str)
        assert isinstance(status_data["created_at"], str)

        # Valid status values
        valid_statuses = {"Q", "R", "D", "F"}
        assert status_data["status"] in valid_statuses

    @staticmethod
    def test_evaluation_with_small_dataset(client: TestClient) -> None:
        """Test evaluation behavior with small datasets."""
        setup_evaluation_test()

        # Load our test datasets to check size
        dataset1_examples = load_test_dataset(f"__rm_-rf__2F_{DATASET_ID_1}.jsonl")
        dataset2_examples = load_test_dataset(f"__rm_-rf__2F_{DATASET_ID_2}.jsonl")

        # Test with both datasets regardless of size
        for dataset_id, _examples in [
            (DATASET_ID_1, dataset1_examples),
            (DATASET_ID_2, dataset2_examples),
        ]:
            payload = create_base_evaluation_payload(dataset_id=dataset_id)
            task_id = start_evaluation_and_get_task_id(client, payload)
            assert_valid_evaluation_status_response(client, task_id)
