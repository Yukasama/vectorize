"""Tests für den Trainings-Endpunkt (/training/train) mit gültigen Daten."""

# import pytest
# from fastapi import status
# from fastapi.testclient import TestClient


# @pytest.mark.training
# class TestTrainingValid:
#     """Tests für den Trainings-Endpunkt (/training/train) mit gültigen Daten."""

#     @staticmethod
#     def test_valid_training(client: TestClient) -> None:
#         """Testet das Training mit gültigen Daten und prüft die Response."""
#         payload = {
#             "model_path": "data/models/localmodel",
#             "dataset_paths": [
#                 "data/dataset_training/classification_test.csv",
#                 "data/dataset_training/classification_test2.csv"
#             ],
#             "output_dir": "data/models/trained_models/test_finetuned_model",
#             "epochs": 1,
#             "learning_rate": 5e-5,
#             "per_device_train_batch_size": 2
#         }
#         response = client.post("/training/train", json=payload)
#         assert response.status_code == status.HTTP_202_ACCEPTED
#         data = response.json()
#         assert data["message"] == "Training started"
#         assert data["model_path"] == payload["model_path"]
