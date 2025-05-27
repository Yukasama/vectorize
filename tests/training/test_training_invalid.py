# ruff: noqa: S101

"""Tests for the training endpoint (/training/train) with invalid data."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient


@pytest.mark.training
class TestTrainingInvalid:
    """Tests for the training endpoint (/training/train) with invalid data."""

    @staticmethod
    def test_invalid_dataset_path(client: TestClient) -> None:
        """Tests training with an invalid dataset path and checks the error response."""
        payload = {
            "model_path": "data/models/localmodel",
            "dataset_paths": [
                "data/datasets/does_not_exist.csv"
            ],
            "output_dir": "data/models/trained_models/my_finetuned_model",
            "epochs": 3,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert any("not found" in str(v).lower() for v in data.values())

    @staticmethod
    def test_invalid_dataset_path_csv(client: TestClient) -> None:
        """Tests training with an invalid dataset path (wrong file extension)."""
        payload = {
            "model_path": "data/models/localmodel",
            "dataset_paths": [
                "data/datasets/__rm_-rf__2F_0a9d5e87-e497-4737-9829-2070780d10df.cs"
            ],
            "output_dir": "data/models/trained_models/my_finetuned_model",
            "epochs": 3,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "all dataset paths must be .csv files" in str(data).lower()
