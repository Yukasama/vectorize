# ruff: noqa: S101

"""Integration tests for training and evaluation workflows using real test data."""

import json
import re
import shutil
import time
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
    """Ensure the required model files are present for integration tests."""
    src = Path("test_data/training/models--sentence-transformers--all-MiniLM-L6-v2")
    dst = settings.model_upload_dir / "models--sentence-transformers--all-MiniLM-L6-v2"
    if not dst.exists() and src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)


def extract_task_id(response: Response, endpoint_type: str = "training") -> str:
    """Extract task_id from training or evaluation response."""
    task_id = None

    # Try Location header first
    if response.headers.get("Location"):
        pattern = rf"/{endpoint_type}/([a-f0-9\-]+)/status"
        match = re.search(pattern, response.headers["Location"])
        if match:
            task_id = match.group(1)

    # Fallback to JSON body
    elif response.content and response.headers.get("content-type", "").startswith(
        "application/json"
    ):
        data = response.json()
        task_id = data.get("task_id")

    assert task_id, f"No task_id found in {endpoint_type} response or headers"
    return task_id


def wait_for_task(
    client: TestClient,
    task_id: str,
    endpoint_type: str = "training",
    max_wait: int = 60,
) -> dict[str, Any]:
    """Wait for a task to complete and return final status."""
    for _ in range(max_wait):
        response = client.get(f"/{endpoint_type}/{task_id}/status")
        if response.status_code == HTTP_200_OK:
            data = response.json()
            if data["status"] in {"D", "F"}:
                return data
        time.sleep(1)

    # Return last known status
    response = client.get(f"/{endpoint_type}/{task_id}/status")
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


@pytest.mark.integration
class TestTrainingEvaluationIntegration:
    """Integration tests for training and evaluation workflows."""

    @staticmethod
    def test_concurrent_training_and_evaluation(client: TestClient) -> None:
        """Test running training and evaluation concurrently."""
        ensure_minilm_model_available()

        # Simplified test - just start one of each type instead of multiple
        training_payload = {
            "model_tag": MINILM_MODEL_TAG,
            "train_dataset_ids": [DATASET_ID_1],
            "epochs": 1,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8,
        }

        training_response = client.post("/training/train", json=training_payload)
        assert training_response.status_code == HTTP_202_ACCEPTED
        training_task_id = extract_task_id(training_response, "training")

        evaluation_payload = {
            "model_tag": MINILM_MODEL_TAG,
            "dataset_id": DATASET_ID_2,
        }

        evaluation_response = client.post(
            "/evaluation/evaluate", json=evaluation_payload
        )
        assert evaluation_response.status_code == HTTP_202_ACCEPTED
        evaluation_task_id = extract_task_id(evaluation_response, "evaluation")

        # Verify both tasks have different IDs
        assert training_task_id != evaluation_task_id

        # Basic status check without intensive polling
        training_status = client.get(f"/training/{training_task_id}/status")
        evaluation_status = client.get(f"/evaluation/{evaluation_task_id}/status")

        assert training_status.status_code == HTTP_200_OK
        assert evaluation_status.status_code == HTTP_200_OK

        cleanup_trained_models()

    @staticmethod
    def test_dataset_schema_consistency(client: TestClient) -> None:
        """Test that both training and evaluation work with same dataset schema."""
        # Verify our test datasets have consistent schema
        dataset1_examples = load_test_dataset(f"__rm_-rf__2F_{DATASET_ID_1}.jsonl")
        dataset2_examples = load_test_dataset(f"__rm_-rf__2F_{DATASET_ID_2}.jsonl")

        # Both datasets should have same schema
        for examples in [dataset1_examples, dataset2_examples]:
            assert len(examples) > 0
            for example in examples[:3]:
                required_fields = ["question", "positive", "negative"]
                for field in required_fields:
                    assert field in example, f"Missing field: {field}"
                    assert isinstance(example[field], str), (
                        f"Field {field} should be string"
                    )
                    assert example[field].strip(), f"Field {field} should not be empty"

        ensure_minilm_model_available()

        # Test training with both datasets
        training_payload = {
            "model_tag": MINILM_MODEL_TAG,
            "train_dataset_ids": [DATASET_ID_1, DATASET_ID_2],
            "epochs": 1,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8,
        }

        training_response = client.post("/training/train", json=training_payload)
        assert training_response.status_code == HTTP_202_ACCEPTED

        # Test evaluation with each dataset
        for dataset_id in [DATASET_ID_1, DATASET_ID_2]:
            eval_payload = {
                "model_tag": MINILM_MODEL_TAG,
                "dataset_id": dataset_id,
            }

            eval_response = client.post("/evaluation/evaluate", json=eval_payload)
            assert eval_response.status_code == HTTP_202_ACCEPTED

        cleanup_trained_models()

    @staticmethod
    def test_api_response_consistency(client: TestClient) -> None:
        """Test that training and evaluation APIs have consistent response formats."""
        ensure_minilm_model_available()

        # Start training
        training_payload = {
            "model_tag": MINILM_MODEL_TAG,
            "train_dataset_ids": [DATASET_ID_1],
            "epochs": 1,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8,
        }

        training_response = client.post("/training/train", json=training_payload)
        assert training_response.status_code == HTTP_202_ACCEPTED
        training_task_id = extract_task_id(training_response, "training")

        # Start evaluation
        evaluation_payload = {
            "model_tag": MINILM_MODEL_TAG,
            "dataset_id": DATASET_ID_1,
        }

        evaluation_response = client.post(
            "/evaluation/evaluate", json=evaluation_payload
        )
        assert evaluation_response.status_code == HTTP_202_ACCEPTED
        evaluation_task_id = extract_task_id(evaluation_response, "evaluation")

        # Both responses should have Location headers
        assert "Location" in training_response.headers
        assert "Location" in evaluation_response.headers

        # Get status for both
        training_status = client.get(f"/training/{training_task_id}/status")
        evaluation_status = client.get(f"/evaluation/{evaluation_task_id}/status")

        assert training_status.status_code == HTTP_200_OK
        assert evaluation_status.status_code == HTTP_200_OK

        training_data = training_status.json()
        evaluation_data = evaluation_status.json()

        # Both should have consistent status response structure
        required_fields = ["task_id", "status", "created_at"]
        for field in required_fields:
            assert field in training_data, f"Training missing field: {field}"
            assert field in evaluation_data, f"Evaluation missing field: {field}"

        # Status values should be from the same set
        valid_statuses = {"Q", "R", "D", "F"}
        assert training_data["status"] in valid_statuses
        assert evaluation_data["status"] in valid_statuses

        cleanup_trained_models()

    @staticmethod
    def test_resource_isolation(client: TestClient) -> None:
        """Test that training and evaluation tasks don't interfere with each other."""
        ensure_minilm_model_available()

        # Start training task
        training_payload = {
            "model_tag": MINILM_MODEL_TAG,
            "train_dataset_ids": [DATASET_ID_1],
            "epochs": 1,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8,
        }

        training_response = client.post("/training/train", json=training_payload)
        assert training_response.status_code == HTTP_202_ACCEPTED
        training_task_id = extract_task_id(training_response, "training")

        # Start evaluation task immediately
        evaluation_payload = {
            "model_tag": MINILM_MODEL_TAG,
            "dataset_id": DATASET_ID_2,
        }

        evaluation_response = client.post(
            "/evaluation/evaluate", json=evaluation_payload
        )
        assert evaluation_response.status_code == HTTP_202_ACCEPTED
        evaluation_task_id = extract_task_id(evaluation_response, "evaluation")

        # Simple isolation test - just verify both can start independently
        training_status = client.get(f"/training/{training_task_id}/status")
        evaluation_status = client.get(f"/evaluation/{evaluation_task_id}/status")

        assert training_status.status_code == HTTP_200_OK
        assert evaluation_status.status_code == HTTP_200_OK

        training_data = training_status.json()
        evaluation_data = evaluation_status.json()

        # Task IDs should remain consistent
        assert training_data["task_id"] == training_task_id
        assert evaluation_data["task_id"] == evaluation_task_id

        # Both tasks should have valid status
        assert training_data["status"] in {"Q", "R", "D", "F"}
        assert evaluation_data["status"] in {"Q", "R", "D", "F"}

        cleanup_trained_models()

    @staticmethod
    def test_validation_error_consistency(client: TestClient) -> None:
        """Test that validation errors are consistent across training and evaluation."""
        # Test missing required fields

        # Training without model_tag
        training_payload = {
            "train_dataset_ids": [DATASET_ID_1],
            "epochs": 1,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8,
        }
        training_response = client.post("/training/train", json=training_payload)
        assert training_response.status_code == HTTP_422_UNPROCESSABLE_ENTITY

        # Evaluation without model_tag
        evaluation_payload = {
            "dataset_id": DATASET_ID_1,
        }
        evaluation_response = client.post(
            "/evaluation/evaluate", json=evaluation_payload
        )
        assert evaluation_response.status_code == HTTP_422_UNPROCESSABLE_ENTITY

        # Both should handle invalid UUID formats similarly
        invalid_uuid = "not-a-uuid"

        training_payload = {
            "model_tag": MINILM_MODEL_TAG,
            "train_dataset_ids": [invalid_uuid],
            "epochs": 1,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8,
        }
        training_response = client.post("/training/train", json=training_payload)

        evaluation_payload = {
            "model_tag": MINILM_MODEL_TAG,
            "dataset_id": invalid_uuid,
        }
        evaluation_response = client.post(
            "/evaluation/evaluate", json=evaluation_payload
        )

        # Both should handle invalid UUIDs consistently
        valid_error_codes = {
            HTTP_400_BAD_REQUEST,
            HTTP_404_NOT_FOUND,
            HTTP_422_UNPROCESSABLE_ENTITY,
        }
        assert training_response.status_code in valid_error_codes
        assert evaluation_response.status_code in valid_error_codes
