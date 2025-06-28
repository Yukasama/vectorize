# ruff: noqa: S101, PLR6301

"""Tests for evaluation service integration with training tasks."""

import asyncio
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.config import settings
from vectorize.dataset.classification import Classification
from vectorize.dataset.dataset_source import DatasetSource
from vectorize.dataset.models import Dataset
from vectorize.dataset.repository import upload_dataset_db
from vectorize.evaluation.schemas import EvaluationRequest
from vectorize.evaluation.service import resolve_evaluation_dataset
from vectorize.training.models import TrainingTask
from vectorize.training.repository import (
    save_training_task,
    update_training_task_validation_dataset,
)

# Test constants
MINILM_MODEL_TAG = "models--sentence-transformers--all-MiniLM-L6-v2"
HTTP_200_OK = status.HTTP_200_OK
HTTP_202_ACCEPTED = status.HTTP_202_ACCEPTED
MAX_RESULTS_SAMPLES = 10  # Constant for magic number
FLOAT_TOLERANCE = 0.001  # Constant for float comparison tolerance


def ensure_test_model_available() -> None:
    """Ensure the required model files are present for integration tests."""
    src = Path("test_data/training/models--sentence-transformers--all-MiniLM-L6-v2")
    dst = settings.model_upload_dir / "models--sentence-transformers--all-MiniLM-L6-v2"
    if not dst.exists() and src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)


def wait_for_evaluation_completion(
    client: TestClient, task_id: str, max_wait: int = 10
) -> dict[str, Any]:
    """Wait for evaluation task to complete and return final status."""
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


def extract_task_id_from_location(location: str) -> str:
    """Extract task ID from Location header."""
    return location.split("/")[-2]  # Extract from "/evaluation/{task_id}/status"


class TestEvaluationIntegration:
    """Test integration between evaluation and training systems."""

    @pytest.mark.asyncio
    async def test_resolve_dataset_with_training_task_id(
        self, session: AsyncSession
    ) -> None:
        """Test resolving dataset using training_task_id."""
        # Create test training task
        task = TrainingTask(
            id=uuid4(),
            model_tag="test-model"
        )
        await save_training_task(session, task)

        # Create validation dataset file
        validation_path = settings.dataset_upload_dir / "validation_test.jsonl"
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        validation_path.write_text(
            '{"question": "test", "positive": "pos", "negative": "neg"}\n'
        )

        # Update task with validation dataset path
        await update_training_task_validation_dataset(
            session, task.id, str(validation_path)
        )

        try:
            # Test request with training_task_id
            request = EvaluationRequest(
                model_tag="test-model",
                training_task_id=str(task.id),
                max_samples=100
            )

            result_path = await resolve_evaluation_dataset(session, request)
            assert result_path == validation_path

        finally:
            # Cleanup
            if validation_path.exists():
                validation_path.unlink()

    @pytest.mark.asyncio
    async def test_resolve_dataset_with_both_ids_fails(
        self, session: AsyncSession
    ) -> None:
        """Test that providing both dataset_id and training_task_id fails."""
        request = EvaluationRequest(
            model_tag="test-model",
            dataset_id=str(uuid4()),
            training_task_id=str(uuid4()),
            max_samples=100
        )

        with pytest.raises(ValueError, match="Cannot specify both"):
            await resolve_evaluation_dataset(session, request)

    @pytest.mark.asyncio
    async def test_resolve_dataset_with_no_ids_fails(
        self, session: AsyncSession
    ) -> None:
        """Test that providing neither dataset_id nor training_task_id fails."""
        request = EvaluationRequest(
            model_tag="test-model",
            max_samples=100
        )

        with pytest.raises(ValueError, match="Must specify either"):
            await resolve_evaluation_dataset(session, request)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_evaluation_metadata_integration(
        self, client: TestClient, session: AsyncSession) -> None:
        """Test end-to-end integration of evaluation metadata functionality."""
        ensure_test_model_available()

        # Create test dataset
        dataset_id = uuid.uuid4()
        dataset_path = settings.dataset_upload_dir / f"{dataset_id}.jsonl"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)

        test_data = [
            {"question": "test", "positive": "pos", "negative": "neg"}
        ]

        with dataset_path.open("w", encoding="utf-8") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        # Save dataset to database
        dataset = Dataset(
            id=dataset_id,
            file_name=f"{dataset_id}.jsonl",
            name=f"Test Dataset {dataset_id}",
            classification=Classification.SENTENCE_TRIPLES,
            rows=len(test_data),
            source=DatasetSource.LOCAL,
        )
        await upload_dataset_db(session, dataset)

        try:
            # Start evaluation
            evaluation_payload = {
                "model_tag": MINILM_MODEL_TAG,
                "dataset_id": str(dataset_id),
                "max_samples": 1
            }

            response = client.post("/evaluation/evaluate", json=evaluation_payload)
            assert response.status_code == HTTP_202_ACCEPTED

            # Extract task ID
            location = response.headers.get("Location")
            assert location is not None
            task_id = extract_task_id_from_location(location)

            # Check immediate status
            immediate_status = client.get(f"/evaluation/{task_id}/status")
            assert immediate_status.status_code == HTTP_200_OK
            immediate_data = immediate_status.json()
            assert immediate_data["task_id"] == task_id

            # Verify metadata fields are present
            metadata_fields = ["model_tag", "dataset_info", "baseline_model_tag"]
            for field in metadata_fields:
                assert field in immediate_data, f"Missing metadata field: {field}"

        finally:
            # Cleanup
            if dataset_path.exists():
                dataset_path.unlink()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_evaluation_service_integration(
        self, client: TestClient, session: AsyncSession) -> None:
        """Test end-to-end integration of the evaluation service."""
        ensure_test_model_available()

        # Create test dataset
        dataset_id = uuid.uuid4()
        dataset_path = settings.dataset_upload_dir / f"{dataset_id}.jsonl"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)

        test_data = [
            {"question": "test", "positive": "pos", "negative": "neg"}
        ]

        with dataset_path.open("w", encoding="utf-8") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        # Save dataset to database
        dataset = Dataset(
            id=dataset_id,
            file_name=f"{dataset_id}.jsonl",
            name=f"Test Dataset {dataset_id}",
            classification=Classification.SENTENCE_TRIPLES,
            rows=len(test_data),
            source=DatasetSource.LOCAL,
        )
        await upload_dataset_db(session, dataset)

        try:
            # Start evaluation
            evaluation_payload = {
                "model_tag": MINILM_MODEL_TAG,
                "dataset_id": str(dataset_id),
                "max_samples": 1
            }

            response = client.post("/evaluation/evaluate", json=evaluation_payload)
            assert response.status_code == HTTP_202_ACCEPTED

            # Extract task ID
            location = response.headers.get("Location")
            assert location is not None
            task_id = extract_task_id_from_location(location)

            # Check status
            status_response = client.get(f"/evaluation/{task_id}/status")
            assert status_response.status_code == HTTP_200_OK

            status_data = status_response.json()
            assert status_data["task_id"] == task_id
            assert status_data["status"] in {"Q", "R", "D", "F"}

        finally:
            # Cleanup
            if dataset_path.exists():
                dataset_path.unlink()


@pytest.mark.evaluation
@pytest.mark.integration
class TestEvaluationMetadataIntegration:
    """Integration tests for evaluation API with metadata functionality."""

    @staticmethod
    def test_evaluation_api_metadata_with_dataset(
        client: TestClient, session: AsyncSession
    ) -> None:
        """Test that evaluation API stores and returns metadata."""

        async def run_test() -> None:
            ensure_test_model_available()

            # Create test dataset
            dataset_id = uuid.uuid4()
            dataset_path = settings.dataset_upload_dir / f"{dataset_id}.jsonl"
            dataset_path.parent.mkdir(parents=True, exist_ok=True)

            test_data = [
                {
                    "question": "What is AI?",
                    "positive": "AI is artificial intelligence",
                    "negative": "AI is not real"
                },
                {
                    "question": "What is ML?",
                    "positive": "ML is machine learning",
                    "negative": "ML is not learning"
                },
            ]

            with dataset_path.open("w", encoding="utf-8") as f:
                for item in test_data:
                    f.write(json.dumps(item) + "\n")

            # Save dataset to database
            dataset = Dataset(
                id=dataset_id,
                file_name=f"{dataset_id}.jsonl",
                name=f"Test Dataset {dataset_id}",
                classification=Classification.SENTENCE_TRIPLES,
                rows=len(test_data),
                source=DatasetSource.LOCAL,
            )
            await upload_dataset_db(session, dataset)

            try:
                # Start evaluation
                evaluation_payload = {
                    "model_tag": MINILM_MODEL_TAG,
                    "dataset_id": str(dataset_id),
                    # Use same model as baseline for test
                    "baseline_model_tag": MINILM_MODEL_TAG,
                    "max_samples": 2
                }

                response = client.post("/evaluation/evaluate", json=evaluation_payload)
                assert response.status_code == HTTP_202_ACCEPTED

                # Extract task ID
                location = response.headers.get("Location")
                assert location is not None
                task_id = extract_task_id_from_location(location)

                # Check immediate status
                immediate_status = client.get(f"/evaluation/{task_id}/status")
                assert immediate_status.status_code == HTTP_200_OK
                immediate_data = immediate_status.json()
                assert immediate_data["task_id"] == task_id

                # Wait briefly and check for metadata
                final_status = wait_for_evaluation_completion(
                    client, task_id, max_wait=5
                )

                # Verify basic response structure
                assert "task_id" in final_status
                assert "status" in final_status
                assert final_status["task_id"] == task_id

            finally:
                # Cleanup
                if dataset_path.exists():
                    dataset_path.unlink()

        # Run the async test
        asyncio.run(run_test())

    @staticmethod
    def test_evaluation_api_response_structure_with_metadata(
        client: TestClient, session: AsyncSession
    ) -> None:
        """Test that evaluation API response includes metadata fields."""

        async def run_test() -> None:
            ensure_test_model_available()

            # Create simple test dataset
            dataset_id = uuid.uuid4()
            dataset_path = settings.dataset_upload_dir / f"{dataset_id}.jsonl"
            dataset_path.parent.mkdir(parents=True, exist_ok=True)

            test_data = [
                {
                    "question": "Test question",
                    "positive": "Positive answer",
                    "negative": "Negative answer"
                }
            ]

            with dataset_path.open("w", encoding="utf-8") as f:
                for item in test_data:
                    f.write(json.dumps(item) + "\n")

            # Save dataset to database
            dataset = Dataset(
                id=dataset_id,
                file_name=f"{dataset_id}.jsonl",
                name=f"Simple Test Dataset {dataset_id}",
                classification=Classification.SENTENCE_TRIPLES,
                rows=len(test_data),
                source=DatasetSource.LOCAL,
            )
            await upload_dataset_db(session, dataset)

            try:
                # Start evaluation
                evaluation_payload = {
                    "model_tag": MINILM_MODEL_TAG,
                    "dataset_id": str(dataset_id),
                    "max_samples": 1
                }

                response = client.post("/evaluation/evaluate", json=evaluation_payload)
                assert response.status_code == HTTP_202_ACCEPTED

                # Extract task ID
                location = response.headers.get("Location")
                assert location is not None
                task_id = extract_task_id_from_location(location)

                # Check status response structure
                status_response = client.get(f"/evaluation/{task_id}/status")
                assert status_response.status_code == HTTP_200_OK

                status_data = status_response.json()

                # Verify required fields are present
                required_fields = [
                    "task_id", "status", "created_at", "updated_at", "progress"
                ]
                for field in required_fields:
                    assert field in status_data, f"Missing required field: {field}"

                # Verify metadata fields are present in schema (even if None)
                metadata_fields = ["model_tag", "dataset_info", "baseline_model_tag"]
                for field in metadata_fields:
                    assert field in status_data, f"Missing metadata field: {field}"

                # Verify task ID is correct
                assert status_data["task_id"] == task_id

            finally:
                # Cleanup
                if dataset_path.exists():
                    dataset_path.unlink()

        # Run the async test
        asyncio.run(run_test())

    @staticmethod
    def test_evaluation_api_concurrent_requests_with_metadata(
        client: TestClient, session: AsyncSession
    ) -> None:
        """Test that concurrent evaluation requests handle metadata correctly."""

        async def run_test() -> None:
            ensure_test_model_available()

            # Create two test datasets
            dataset_ids = [uuid.uuid4(), uuid.uuid4()]
            dataset_paths = []

            test_data = [
                {
                    "question": "Concurrent test",
                    "positive": "Good answer",
                    "negative": "Bad answer"
                }
            ]

            for i, dataset_id in enumerate(dataset_ids):
                dataset_path = settings.dataset_upload_dir / f"{dataset_id}.jsonl"
                dataset_path.parent.mkdir(parents=True, exist_ok=True)
                dataset_paths.append(dataset_path)

                with dataset_path.open("w", encoding="utf-8") as f:
                    for item in test_data:
                        f.write(json.dumps(item) + "\n")

                # Save dataset to database
                dataset = Dataset(
                    id=dataset_id,
                    file_name=f"{dataset_id}.jsonl",
                    name=f"Concurrent Test Dataset {i + 1}",
                    classification=Classification.SENTENCE_TRIPLES,
                    rows=len(test_data),
                    source=DatasetSource.LOCAL,
                )
                await upload_dataset_db(session, dataset)

            try:
                # Start two evaluations concurrently
                task_ids = []
                for dataset_id in dataset_ids:
                    evaluation_payload = {
                        "model_tag": MINILM_MODEL_TAG,
                        "dataset_id": str(dataset_id),
                        "max_samples": 1
                    }

                    response = client.post(
                        "/evaluation/evaluate", json=evaluation_payload
                    )
                    assert response.status_code == HTTP_202_ACCEPTED

                    location = response.headers.get("Location")
                    assert location is not None
                    task_id = extract_task_id_from_location(location)
                    task_ids.append(task_id)

                # Verify both tasks have different IDs
                assert task_ids[0] != task_ids[1]

                # Verify both have valid status responses
                for task_id in task_ids:
                    status_response = client.get(f"/evaluation/{task_id}/status")
                    assert status_response.status_code == HTTP_200_OK

                    status_data = status_response.json()
                    assert status_data["task_id"] == task_id
                    assert status_data["status"] in {"Q", "R", "D", "F"}

            finally:
                # Cleanup
                for dataset_path in dataset_paths:
                    if dataset_path.exists():
                        dataset_path.unlink()

        # Run the async test
        asyncio.run(run_test())
