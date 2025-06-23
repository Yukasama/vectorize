# ruff: noqa: S101

"""Tests for the training cancel endpoint."""

import shutil
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from vectorize.config import settings

MINILM_MODEL_TAG = "models--sentence-transformers--all-MiniLM-L6-v2"
DATASET_ID_1 = "0b30b284-f7fe-4e6c-a270-17cafc5b5bcb"
DEFAULT_EPOCHS = 1
DEFAULT_LR = 0.00005
DEFAULT_BATCH_SIZE = 8
REDIS_EXPIRATION_SECONDS = 3600

HTTP_200_OK = status.HTTP_200_OK
HTTP_202_ACCEPTED = status.HTTP_202_ACCEPTED
HTTP_400_BAD_REQUEST = status.HTTP_400_BAD_REQUEST
HTTP_404_NOT_FOUND = status.HTTP_404_NOT_FOUND


def ensure_minilm_model_available() -> None:
    """Ensure the required model files are present for training tests."""
    src = Path("test_data/training/models--sentence-transformers--all-MiniLM-L6-v2")
    dst = settings.model_upload_dir / "models--sentence-transformers--all-MiniLM-L6-v2"
    if not dst.exists() and src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)


@pytest.mark.training
class TestTrainingCancel:
    """Tests for the training cancel endpoint."""

    @staticmethod
    def test_cancel_non_existent_task(client: TestClient) -> None:
        """Test canceling a training task that doesn't exist."""
        random_task_id = str(uuid.uuid4())
        response = client.post(f"/training/{random_task_id}/cancel")

        assert response.status_code == HTTP_404_NOT_FOUND
        data = response.json()
        assert "not found" in str(data).lower()

    @staticmethod
    @patch("redis.Redis")
    def test_cancel_running_task(mock_redis_class: Mock, client: TestClient) -> None:
        """Test canceling a running training task."""
        ensure_minilm_model_available()

        # Mock Redis client
        mock_redis = Mock()
        mock_redis_class.return_value = mock_redis

        # Start a training task first
        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "train_dataset_ids": [DATASET_ID_1],
            "epochs": DEFAULT_EPOCHS,
            "learning_rate": DEFAULT_LR,
            "per_device_train_batch_size": DEFAULT_BATCH_SIZE,
        }

        response = client.post("/training/train", json=payload)
        assert response.status_code == HTTP_202_ACCEPTED

        # Extract task_id from Location header
        location = response.headers.get("Location")
        assert location is not None
        task_id = location.split("/")[-2]  # Extract from /training/{task_id}/status

        # Now cancel the task
        cancel_response = client.post(f"/training/{task_id}/cancel")
        assert cancel_response.status_code == HTTP_200_OK

        # Verify Redis set was called for cancellation signal
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert call_args[0][0].startswith("cancel_training:")  # cancellation key
        assert call_args[0][1] == "true"  # value
        assert call_args[1]["ex"] == REDIS_EXPIRATION_SECONDS  # expiration

        # Verify task status is now CANCELED
        status_response = client.get(f"/training/{task_id}/status")
        assert status_response.status_code == HTTP_200_OK
        status_data = status_response.json()
        assert status_data["status"] == "C"  # TaskStatus.CANCELED.value
        assert "canceled by user" in status_data.get("error_msg", "").lower()

    @staticmethod
    def test_cancel_already_completed_task(client: TestClient) -> None:
        """Test trying to cancel a task that's already completed."""
        ensure_minilm_model_available()

        # Start a training task first
        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "train_dataset_ids": [DATASET_ID_1],
            "epochs": DEFAULT_EPOCHS,
            "learning_rate": DEFAULT_LR,
            "per_device_train_batch_size": DEFAULT_BATCH_SIZE,
        }

        response = client.post("/training/train", json=payload)
        assert response.status_code == HTTP_202_ACCEPTED

        # Extract task_id
        location = response.headers.get("Location")
        assert location is not None
        task_id = location.split("/")[-2]

        # First cancel should work
        cancel_response = client.post(f"/training/{task_id}/cancel")
        assert cancel_response.status_code == HTTP_200_OK

        # Second cancel should fail with 400 Bad Request
        second_cancel_response = client.post(f"/training/{task_id}/cancel")
        assert second_cancel_response.status_code == HTTP_400_BAD_REQUEST
        assert "cannot cancel task with status" in second_cancel_response.text.lower()

    @staticmethod
    @patch("redis.Redis")
    def test_cancel_with_redis_failure(
        mock_redis_class: Mock, client: TestClient
    ) -> None:
        """Test canceling when Redis connection fails."""
        ensure_minilm_model_available()

        # Mock Redis to raise an exception
        mock_redis = Mock()
        mock_redis.set.side_effect = Exception("Redis connection failed")
        mock_redis_class.return_value = mock_redis

        # Start a training task first
        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "train_dataset_ids": [DATASET_ID_1],
            "epochs": DEFAULT_EPOCHS,
            "learning_rate": DEFAULT_LR,
            "per_device_train_batch_size": DEFAULT_BATCH_SIZE,
        }

        response = client.post("/training/train", json=payload)
        assert response.status_code == HTTP_202_ACCEPTED

        # Extract task_id
        location = response.headers.get("Location")
        assert location is not None
        task_id = location.split("/")[-2]

        # Cancel should still work (gracefully handle Redis failure)
        cancel_response = client.post(f"/training/{task_id}/cancel")
        assert cancel_response.status_code == HTTP_200_OK

        status_response = client.get(f"/training/{task_id}/status")
        assert status_response.status_code == HTTP_200_OK
        status_data = status_response.json()
        assert status_data["status"] == "C"  # TaskStatus.CANCELED.value
