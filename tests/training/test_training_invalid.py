# ruff: noqa: S101

"""Tests for the training endpoint (/training/train) with invalid data."""

import asyncio
import uuid
from pathlib import Path

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.common.task_status import TaskStatus
from vectorize.training.models import TrainingTask


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

    @staticmethod
    def test_invalid_model_path(client: TestClient) -> None:
        """Tests training with an invalid model_path and checks the error response."""
        payload = {
            "model_path": "data/models/does_not_exist",
            "dataset_paths": [
                "data/datasets/__rm_-rf__2F_0b30b284-f7fe-4e6c-a270-17cafc5b5bcb.csv",
                "data/datasets/__rm_-rf__2F_0a9d5e87-e497-4737-9829-2070780d10df.csv"
            ],
            "output_dir": "data/models/trained_models/should_not_exist",
            "epochs": 3,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "model path" in str(data).lower() or "not found" in str(data).lower()

    @staticmethod
    def test_empty_dataset_list(client: TestClient) -> None:
        """Tests training with an empty dataset_paths list."""
        payload = {
            "model_path": "data/models/localmodel",
            "dataset_paths": [],
            "output_dir": "data/models/trained_models/should_not_exist",
            "epochs": 3,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code in {422, 400}
        data = response.json()
        assert "dataset" in str(data).lower() or "empty" in str(data).lower()

    @staticmethod
    def test_negative_epochs(client: TestClient) -> None:
        """Tests training with a negative number of epochs (should fail validation)."""
        payload = {
            "model_path": "data/models/localmodel",
            "dataset_paths": [
                "data/datasets/__rm_-rf__2F_0a9d5e87-e497-4737-9829-2070780d10df.csv"
            ],
            "output_dir": "data/models/trained_models/should_not_exist",
            "epochs": -5,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "epochs" in str(data).lower() or "negative" in str(data).lower()

    @staticmethod
    def test_zero_batch_size(client: TestClient) -> None:
        """Tests training with a zero batch size (should fail validation)."""
        payload = {
            "model_path": "data/models/localmodel",
            "dataset_paths": [
                "data/datasets/__rm_-rf__2F_0a9d5e87-e497-4737-9829-2070780d10df.csv"
            ],
            "output_dir": "data/models/trained_models/should_not_exist",
            "epochs": 3,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 0
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "batch" in str(data).lower() or "zero" in str(data).lower()

    @staticmethod
    def test_negative_learning_rate(client: TestClient) -> None:
        """Tests training with a negative learning rate (should fail validation)."""
        payload = {
            "model_path": "data/models/localmodel",
            "dataset_paths": [
                "data/datasets/__rm_-rf__2F_0a9d5e87-e497-4737-9829-2070780d10df.csv"
            ],
            "output_dir": "data/models/trained_models/should_not_exist",
            "epochs": 3,
            "learning_rate": -0.01,
            "per_device_train_batch_size": 8
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "learning" in str(data).lower() or "negative" in str(data).lower()

    @staticmethod
    @pytest.mark.asyncio
    async def test_task_status_failed_runtime(
        client: TestClient, session: AsyncSession, tmp_path: Path
    ) -> None:
        """Test that a training task fails at runtime (invalid CSV content).

        and status is FAILED in DB.
        """
        bad_csv = tmp_path / "empty.csv"
        bad_csv.write_text("")
        payload = {
            "model_path": "data/models/localmodel",
            "dataset_paths": [str(bad_csv)],
            "output_dir": str(tmp_path / "should_not_exist"),
            "epochs": 3,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8,
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code == status.HTTP_202_ACCEPTED
        task_id = uuid.UUID(response.json()["task_id"])

        for _ in range(20):
            session.expire_all()
            task = await session.get(TrainingTask, task_id)
            if task and task.task_status == TaskStatus.FAILED:
                break
            await asyncio.sleep(0.5)
        else:
            raise AssertionError("Task did not reach FAILED status in time")

        assert task.error_msg
        err = task.error_msg.lower()
        assert any(
            substring in err
            for substring in [
                "csv",
                "empty",
                "failed",
                "datasetgenerationerror",
                "generating the dataset",
            ]
        )

    @staticmethod
    def test_get_training_status_not_found(client: TestClient) -> None:
        """Tests the GET /training/{task_id}/status endpoint with an invalid task ID."""
        invalid_task_id = str(uuid.uuid4())
        response = client.get(f"/training/{invalid_task_id}/status")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "not found" in str(data).lower()
