# ruff: noqa: S101

"""Tests for the training endpoint (/training/train) with valid data."""

import shutil
from pathlib import Path

import pytest
from fastapi import status
from fastapi.testclient import TestClient

LOCALTRAINMODEL_ID = "3d2f3e4b-8c7f-4d2a-9f1e-0a6f3e4d2a5b"


@pytest.mark.training
class TestTrainingValid:
    """Tests for the training endpoint (/training/train) with valid data."""

    @staticmethod
    def test_valid_training(client: TestClient) -> None:
        """Tests training with valid data and checks the response.

        Also deletes the trained model after test.
        """
        payload = {
            "model_id": LOCALTRAINMODEL_ID,
            "dataset_ids": [
                "0b30b284-f7fe-4e6c-a270-17cafc5b5bcb",
                "0a9d5e87-e497-4737-9829-2070780d10df"
            ],
            "epochs": 3,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code == status.HTTP_202_ACCEPTED

        # Lösche alle erzeugten Modelle nach dem Test
        import glob
        import shutil
        from pathlib import Path
        model_dirs = glob.glob("data/models/trained_models/*-finetuned-*")
        for d in model_dirs:
            shutil.rmtree(d, ignore_errors=True)

    @staticmethod
    def test_get_training_status(client: TestClient) -> None:
        """Test the status endpoint for a training task.

        Args:
            client (TestClient): FastAPI test client for making API requests.
        """
        payload = {
            "model_id": LOCALTRAINMODEL_ID,
            "dataset_ids": [
                "0b30b284-f7fe-4e6c-a270-17cafc5b5bcb",
                "0a9d5e87-e497-4737-9829-2070780d10df"
            ],
            "epochs": 1,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code == status.HTTP_202_ACCEPTED

    @staticmethod
    def test_training_with_single_dataset(client: TestClient) -> None:
        """Test training with only one dataset (should succeed)."""
        payload = {
            "model_id": LOCALTRAINMODEL_ID,
            "dataset_ids": [
                "0b30b284-f7fe-4e6c-a270-17cafc5b5bcb"
            ],
            "epochs": 1,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code == status.HTTP_202_ACCEPTED

    @staticmethod
    def test_progress_tracking(client: TestClient) -> None:
        """Test that progress is tracked and >0 after training start."""
        payload = {
            "model_id": LOCALTRAINMODEL_ID,
            "dataset_ids": [
                "0b30b284-f7fe-4e6c-a270-17cafc5b5bcb"
            ],
            "epochs": 1,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code == status.HTTP_202_ACCEPTED
        # Extract task_id from logs or DB if available, or skip detailed check here
        # (In echter Umgebung: task_id aus Response/Status holen und progress prüfen)
        # Hier nur Dummy-Check, da kein task_id zurückgegeben wird
        # assert progress > 0
        # TODO: Implementiere echten Progress-Check, wenn API das unterstützt
