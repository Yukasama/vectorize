# ruff: noqa: S101

"""Tests for the training endpoint (/training/train) with valid data."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient


@pytest.mark.training
class TestTrainingValid:
    """Tests for the training endpoint (/training/train) with valid data."""

    @staticmethod
    def test_valid_training(client: TestClient) -> None:
        """Tests training with valid data and checks the response."""
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
