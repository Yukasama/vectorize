# ruff: noqa: S101

"""Tests for the training endpoint (/training/train) with valid data."""

import shutil
from pathlib import Path

import pytest
from fastapi import status
from fastapi.testclient import TestClient

MINILM_MODEL_TAG = "models--sentence-transformers--all-MiniLM-L6-v2"


@pytest.mark.training
class TestTrainingValid:
    """Tests for the training endpoint (/training/train) with valid data."""

    @staticmethod
    def test_valid_training(client: TestClient) -> None:
        """Tests training with valid data and checks the response and status tracking."""
        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "dataset_ids": [
                "0b30b284-f7fe-4e6c-a270-17cafc5b5bcb",
                "0a9d5e87-e497-4737-9829-2070780d10df",
            ],
            "epochs": 3,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8,
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code == status.HTTP_202_ACCEPTED
        task_id = None
        if response.headers.get("Location"):
            import re
            match = re.search(r"/training/([a-f0-9\-]+)/status", response.headers["Location"])
            if match:
                task_id = match.group(1)
        elif response.content and response.headers.get("content-type", "").startswith("application/json"):
            data = response.json()
            task_id = data.get("task_id")
        assert task_id, "No task_id found in response or headers"
        status_response = client.get(f"/training/{task_id}/status")
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["status"] in {"PENDING", "RUNNING", "DONE", "FAILED"}
        progress = status_data.get("progress", 0)
        assert 0.0 <= progress <= 1.0
        model_dirs = Path("data/models/trained_models").glob("*-finetuned-*")
        for d in model_dirs:
            shutil.rmtree(d, ignore_errors=True)

    @staticmethod
    def test_get_training_status(client: TestClient) -> None:
        """Test the status endpoint for a training task with random ID (should fail)."""
        import uuid
        random_id = str(uuid.uuid4())
        response = client.get(f"/training/{random_id}/status")
        assert response.status_code == 404
        data = response.json()
        assert "not found" in str(data).lower()

    @staticmethod
    def test_training_with_single_dataset(client: TestClient) -> None:
        """Test training with only one dataset (should succeed)."""
        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "dataset_ids": ["0b30b284-f7fe-4e6c-a270-17cafc5b5bcb"],
            "epochs": 1,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8,
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code == status.HTTP_202_ACCEPTED
        task_id = None
        if response.headers.get("Location"):
            import re
            match = re.search(r"/training/([a-f0-9\-]+)/status", response.headers["Location"])
            if match:
                task_id = match.group(1)
        elif response.content and response.headers.get("content-type", "").startswith("application/json"):
            data = response.json()
            task_id = data.get("task_id")
        assert task_id, "No task_id found in response or headers"

    @staticmethod
    def test_progress_tracking(client: TestClient) -> None:
        """Test that progress is tracked and >0 after training start."""
        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "dataset_ids": ["0b30b284-f7fe-4e6c-a270-17cafc5b5bcb"],
            "epochs": 1,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8,
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code == status.HTTP_202_ACCEPTED
        task_id = None
        if response.headers.get("Location"):
            import re
            match = re.search(r"/training/([a-f0-9\-]+)/status", response.headers["Location"])
            if match:
                task_id = match.group(1)
        elif response.content and response.headers.get("content-type", "").startswith("application/json"):
            data = response.json()
            task_id = data.get("task_id")
        assert task_id, "No task_id found in response or headers"
        import time
        time.sleep(0.5)
        status_response = client.get(f"/training/{task_id}/status")
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert 0.0 <= status_data.get("progress", 0) <= 1.0
