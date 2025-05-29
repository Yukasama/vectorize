# ruff: noqa: S101

"""Tests for the training endpoint (/training/train) with valid data."""

import shutil
from pathlib import Path

import pytest
from fastapi import status
from fastapi.testclient import TestClient


@pytest.mark.training
class TestTrainingValid:
    """Tests for the training endpoint (/training/train) with valid data."""

    @staticmethod
    def test_valid_training(client: TestClient) -> None:
        """Tests training with valid data and checks the response.

        Also deletes the trained model after test.
        """
        payload = {
            "model_path": "data/models/localmodel",
            "dataset_paths": [
                "data/datasets/__rm_-rf__2F_0b30b284-f7fe-4e6c-a270-17cafc5b5bcb.csv",
                "data/datasets/__rm_-rf__2F_0a9d5e87-e497-4737-9829-2070780d10df.csv"
            ],
            "output_dir": "data/models/trained_models/my_finetuned_model",
            "epochs": 3,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code == status.HTTP_202_ACCEPTED
        data = response.json()
        assert data["message"] == "Training started"
        assert data["model_path"] == payload["model_path"]

        trained_model_dir = Path(payload["output_dir"])
        if trained_model_dir.exists() and trained_model_dir.is_dir():
            shutil.rmtree(trained_model_dir)

    @staticmethod
    def test_get_training_status(client: TestClient) -> None:
        """Tests the GET /training/{task_id}/status endpoint with a valid task ID."""
        # Starte ein Training, um eine gültige Task-ID zu erhalten
        payload = {
            "model_path": "data/models/localmodel",
            "dataset_paths": [
                "data/datasets/__rm_-rf__2F_0b30b284-f7fe-4e6c-a270-17cafc5b5bcb.csv",
                "data/datasets/__rm_-rf__2F_0a9d5e87-e497-4737-9829-2070780d10df.csv"
            ],
            "output_dir": "data/models/trained_models/my_finetuned_model_status_test",
            "epochs": 1,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code == status.HTTP_202_ACCEPTED
        data = response.json()
        task_id = data["task_id"]

        # Rufe den Status-Endpunkt ab
        status_response = client.get(f"/training/{task_id}/status")
        assert status_response.status_code == status.HTTP_200_OK
        status_data = status_response.json()
        assert status_data["task_id"] == task_id
        assert status_data["status"] in {"PENDING", "DONE", "FAILED", "CANCELED"}
        assert status_data["created_at"] is not None
        # Enddatum und error_msg können None sein, je nach Status
        # Aufräumen
        trained_model_dir = Path(payload["output_dir"])
        if trained_model_dir.exists() and trained_model_dir.is_dir():
            shutil.rmtree(trained_model_dir)
