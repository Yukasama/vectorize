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
    from vectorize.config import settings

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
        match = re.search(r"/evaluation/([a-f0-9\-]+)/status", response.headers["Location"])
        if match:
            task_id = match.group(1)

    # Fallback to JSON body
    elif response.content and response.headers.get("content-type", "").startswith("application/json"):
        data = response.json()
        task_id = data.get("task_id")

    assert task_id, "No task_id found in response or headers"
    return task_id


def wait_for_evaluation_completion(client: TestClient, task_id: str, max_wait: int = 60) -> dict[str, Any]:
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
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    assert examples, f"No examples found in {file_path}"
    return examples


@pytest.mark.evaluation
class TestEvaluationComprehensive:
    """Comprehensive tests for evaluation endpoints with real test data."""

    def test_evaluation_different_datasets(self, client: TestClient) -> None:
        """Test evaluation with different datasets."""
        ensure_minilm_model_available()

        # Test with first dataset
        payload1 = {
            "model_tag": MINILM_MODEL_TAG,
            "dataset_id": DATASET_ID_1,
        }

        response1 = client.post("/evaluation/evaluate", json=payload1)
        assert response1.status_code == HTTP_202_ACCEPTED
        task_id1 = extract_task_id_from_response(response1)

        # Test with second dataset
        payload2 = {
            "model_tag": MINILM_MODEL_TAG,
            "dataset_id": DATASET_ID_2,
        }

        response2 = client.post("/evaluation/evaluate", json=payload2)
        assert response2.status_code == HTTP_202_ACCEPTED
        task_id2 = extract_task_id_from_response(response2)

        # Both should have different task IDs
        assert task_id1 != task_id2

        # Both should have valid status
        status1 = client.get(f"/evaluation/{task_id1}/status")
        status2 = client.get(f"/evaluation/{task_id2}/status")
        assert status1.status_code == HTTP_200_OK
        assert status2.status_code == HTTP_200_OK

    def test_evaluation_nonexistent_model(self, client: TestClient) -> None:
        """Test evaluation with nonexistent model tag."""
        payload = {
            "model_tag": "nonexistent-model-tag",
            "dataset_id": DATASET_ID_1,
        }

        response = client.post("/evaluation/evaluate", json=payload)
        # Should either fail immediately or start and fail during execution
        assert response.status_code in [HTTP_400_BAD_REQUEST, HTTP_202_ACCEPTED]

    def test_evaluation_nonexistent_dataset(self, client: TestClient) -> None:
        """Test evaluation with nonexistent dataset ID."""
        ensure_minilm_model_available()

        fake_dataset_id = str(uuid.uuid4())
        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "dataset_id": fake_dataset_id,
        }

        response = client.post("/evaluation/evaluate", json=payload)
        # Should either fail immediately or start and fail during execution
        assert response.status_code in [HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND, HTTP_202_ACCEPTED]

    def test_evaluation_status_invalid_task_id(self, client: TestClient) -> None:
        """Test status endpoint with invalid task IDs."""
        # Test with completely random UUID
        random_id = str(uuid.uuid4())
        response = client.get(f"/evaluation/{random_id}/status")
        assert response.status_code == HTTP_404_NOT_FOUND

        # Test with invalid UUID format
        response = client.get("/evaluation/invalid-uuid/status")
        assert response.status_code in [HTTP_400_BAD_REQUEST, HTTP_422_UNPROCESSABLE_ENTITY]

    def test_evaluation_response_structure(self, client: TestClient) -> None:
        """Test that evaluation response has correct structure."""
        ensure_minilm_model_available()

        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "dataset_id": DATASET_ID_1,
        }

        response = client.post("/evaluation/evaluate", json=payload)
        assert response.status_code == HTTP_202_ACCEPTED

        # Check Location header
        assert "Location" in response.headers
        location = response.headers["Location"]
        assert "/evaluation/" in location
        assert "/status" in location

        # Extract and validate task ID from location
        task_id = extract_task_id_from_response(response)
        assert uuid.UUID(task_id)  # Should be valid UUID

    def test_evaluation_status_response_structure(self, client: TestClient) -> None:
        """Test that status response has correct structure."""
        ensure_minilm_model_available()

        # Start an evaluation task
        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "dataset_id": DATASET_ID_1,
        }

        response = client.post("/evaluation/evaluate", json=payload)
        assert response.status_code == HTTP_202_ACCEPTED
        task_id = extract_task_id_from_response(response)

        # Check status response structure
        status_response = client.get(f"/evaluation/{task_id}/status")
        assert status_response.status_code == HTTP_200_OK

        status_data = status_response.json()
        required_fields = ["task_id", "status", "created_at"]
        for field in required_fields:
            assert field in status_data, f"Missing required field: {field}"

        assert status_data["task_id"] == task_id
        assert status_data["status"] in {"Q", "R", "D", "F"}

        # created_at should be ISO format timestamp
        assert isinstance(status_data["created_at"], str)

    def test_evaluation_results_endpoint(self, client: TestClient) -> None:
        """Test evaluation results endpoint."""
        ensure_minilm_model_available()

        # Start evaluation
        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "dataset_id": DATASET_ID_1,
        }

        response = client.post("/evaluation/evaluate", json=payload)
        assert response.status_code == HTTP_202_ACCEPTED
        task_id = extract_task_id_from_response(response)

        # Try to get results (may not be ready yet)
        results_response = client.get(f"/evaluation/{task_id}/results")
        # Should either be 404 (not ready) or 200 (ready)
        assert results_response.status_code in [HTTP_200_OK, HTTP_404_NOT_FOUND]

        if results_response.status_code == HTTP_200_OK:
            results_data = results_response.json()
            # Check that results have expected structure
            assert isinstance(results_data, dict)
            # Results structure may vary, but should be JSON serializable

    def test_evaluation_dataset_schema_validation(self, client: TestClient) -> None:
        """Test that evaluation works with our test data schema (question/positive/negative)."""
        # Verify our test datasets have the correct schema
        dataset1_examples = load_test_dataset(f"__rm_-rf__2F_{DATASET_ID_2}.jsonl")
        dataset2_examples = load_test_dataset(f"__rm_-rf__2F_{DATASET_ID_1}.jsonl")

        # Check that examples have required fields for evaluation
        for examples in [dataset1_examples, dataset2_examples]:
            assert len(examples) > 0, "Dataset should not be empty"

            for example in examples[:3]:  # Check first few examples
                assert "question" in example, "Missing 'question' field"
                assert "positive" in example, "Missing 'positive' field"
                assert "negative" in example, "Missing 'negative' field"

                # Validate field types
                assert isinstance(example["question"], str), "'question' should be string"
                assert isinstance(example["positive"], str), "'positive' should be string"
                assert isinstance(example["negative"], str), "'negative' should be string"

                # Validate content is not empty
                assert example["question"].strip(), "'question' should not be empty"
                assert example["positive"].strip(), "'positive' should not be empty"
                assert example["negative"].strip(), "'negative' should not be empty"

    def test_evaluation_concurrent_requests(self, client: TestClient) -> None:
        """Test handling of concurrent evaluation requests."""
        ensure_minilm_model_available()

        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "dataset_id": DATASET_ID_1,
        }

        # Start first evaluation
        response1 = client.post("/evaluation/evaluate", json=payload)
        assert response1.status_code == HTTP_202_ACCEPTED
        task_id1 = extract_task_id_from_response(response1)

        # Start second evaluation immediately
        payload2 = {
            "model_tag": MINILM_MODEL_TAG,
            "dataset_id": DATASET_ID_2,
        }
        response2 = client.post("/evaluation/evaluate", json=payload2)
        assert response2.status_code == HTTP_202_ACCEPTED
        task_id2 = extract_task_id_from_response(response2)

        # Both should have different task IDs
        assert task_id1 != task_id2

        # Both should have valid status
        status1 = client.get(f"/evaluation/{task_id1}/status")
        status2 = client.get(f"/evaluation/{task_id2}/status")
        assert status1.status_code == HTTP_200_OK
        assert status2.status_code == HTTP_200_OK

    def test_evaluation_missing_required_fields(self, client: TestClient) -> None:
        """Test evaluation request with missing required fields."""
        # Missing model_tag
        payload = {
            "dataset_id": DATASET_ID_1,
        }
        response = client.post("/evaluation/evaluate", json=payload)
        assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY

        # Missing dataset_id - backend accepts this and fails in background task
        payload = {
            "model_tag": MINILM_MODEL_TAG,
        }
        response = client.post("/evaluation/evaluate", json=payload)
        assert response.status_code == HTTP_202_ACCEPTED

    def test_evaluation_invalid_request_data(self, client: TestClient) -> None:
        """Test evaluation with invalid request data types."""
        # Invalid model_tag type
        payload = {
            "model_tag": 123,  # Should be string
            "dataset_id": DATASET_ID_1,
        }
        response = client.post("/evaluation/evaluate", json=payload)
        assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY

        # Invalid dataset_id type
        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "dataset_id": 123,  # Should be string
        }
        response = client.post("/evaluation/evaluate", json=payload)
        assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY

    def test_evaluation_empty_string_parameters(self, client: TestClient) -> None:
        """Test evaluation with empty string parameters."""
        # Empty model_tag - backend accepts this and fails in background task
        payload = {
            "model_tag": "",
            "dataset_id": DATASET_ID_1,
        }
        response = client.post("/evaluation/evaluate", json=payload)
        assert response.status_code == HTTP_202_ACCEPTED

        # Empty dataset_id - backend also accepts this and fails in background task
        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "dataset_id": "",
        }
        response = client.post("/evaluation/evaluate", json=payload)
        assert response.status_code == HTTP_202_ACCEPTED

    def test_evaluation_status_field_validation(self, client: TestClient) -> None:
        """Test that evaluation status contains all expected fields."""
        ensure_minilm_model_available()

        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "dataset_id": DATASET_ID_1,
        }

        response = client.post("/evaluation/evaluate", json=payload)
        assert response.status_code == HTTP_202_ACCEPTED
        task_id = extract_task_id_from_response(response)

        status_response = client.get(f"/evaluation/{task_id}/status")
        assert status_response.status_code == HTTP_200_OK
        status_data = status_response.json()

        # Required fields
        required_fields = ["task_id", "status", "created_at"]
        for field in required_fields:
            assert field in status_data, f"Missing required field: {field}"

        # Field type validation
        assert isinstance(status_data["task_id"], str)
        assert isinstance(status_data["status"], str)
        assert isinstance(status_data["created_at"], str)

        # Valid status values
        valid_statuses = {"Q", "R", "D", "F"}
        assert status_data["status"] in valid_statuses

    def test_evaluation_with_small_dataset(self, client: TestClient) -> None:
        """Test evaluation behavior with small datasets."""
        ensure_minilm_model_available()

        # Load our test datasets to check size
        dataset1_examples = load_test_dataset(f"__rm_-rf__2F_{DATASET_ID_1}.jsonl")
        dataset2_examples = load_test_dataset(f"__rm_-rf__2F_{DATASET_ID_2}.jsonl")

        # Test with both datasets regardless of size
        for dataset_id, examples in [(DATASET_ID_1, dataset1_examples), (DATASET_ID_2, dataset2_examples)]:
            payload = {
                "model_tag": MINILM_MODEL_TAG,
                "dataset_id": dataset_id,
            }

            response = client.post("/evaluation/evaluate", json=payload)
            assert response.status_code == HTTP_202_ACCEPTED

            task_id = extract_task_id_from_response(response)
            status_response = client.get(f"/evaluation/{task_id}/status")
            assert status_response.status_code == HTTP_200_OK
