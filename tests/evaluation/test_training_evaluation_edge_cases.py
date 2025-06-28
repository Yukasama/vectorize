# ruff: noqa: S101

"""Edge case and stress tests for training and evaluation endpoints."""

import json
import re
import shutil
import uuid
from pathlib import Path

import pytest
from fastapi import status
from fastapi.testclient import TestClient

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
HTTP_500_INTERNAL_SERVER_ERROR = status.HTTP_500_INTERNAL_SERVER_ERROR

UUID_LENGTH = 36
UUID_HYPHEN_COUNT = 4


def ensure_minilm_model_available() -> None:
    """Ensure the required model files are present for edge case tests."""
    src = Path("test_data/training/models--sentence-transformers--all-MiniLM-L6-v2")
    dst = settings.model_upload_dir / "models--sentence-transformers--all-MiniLM-L6-v2"
    if not dst.exists() and src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)


def cleanup_trained_models() -> None:
    """Clean up any trained model directories."""
    trained_models_dir = settings.model_upload_dir / "trained_models"
    if trained_models_dir.exists():
        for model_dir in trained_models_dir.glob("*-finetuned-*"):
            shutil.rmtree(model_dir, ignore_errors=True)


@pytest.mark.edge_cases
class TestTrainingEdgeCases:
    """Edge case tests for training endpoints."""

    @staticmethod
    def test_training_extreme_parameters(client: TestClient) -> None:
        """Test training with extreme but potentially valid parameters."""
        ensure_minilm_model_available()

        test_cases = [
            {
                "name": "very_small_learning_rate",
                "payload": {
                    "model_tag": MINILM_MODEL_TAG,
                    "train_dataset_ids": [DATASET_ID_1],
                    "epochs": 1,
                    "learning_rate": 1e-8,  # Extremely small
                    "per_device_train_batch_size": 1,
                },
            },
            {
                "name": "large_batch_size",
                "payload": {
                    "model_tag": MINILM_MODEL_TAG,
                    "train_dataset_ids": [DATASET_ID_1],
                    "epochs": 1,
                    "learning_rate": 0.00005,
                    "per_device_train_batch_size": 128,  # Large batch
                },
            },
            {
                "name": "many_epochs",
                "payload": {
                    "model_tag": MINILM_MODEL_TAG,
                    "train_dataset_ids": [DATASET_ID_1],
                    "epochs": 100,  # Many epochs (may be slow)
                    "learning_rate": 0.00005,
                    "per_device_train_batch_size": 8,
                },
            },
        ]

        for test_case in test_cases:
            response = client.post("/training/train", json=test_case["payload"])
            # Should either accept or reject with proper error code
            assert response.status_code in {
                HTTP_202_ACCEPTED,
                HTTP_400_BAD_REQUEST,
                HTTP_422_UNPROCESSABLE_ENTITY,
            }, f"Unexpected status for {test_case['name']}: {response.status_code}"

        cleanup_trained_models()

    @staticmethod
    def test_training_boundary_values(client: TestClient) -> None:
        """Test training with boundary values for parameters."""
        ensure_minilm_model_available()

        boundary_tests = [
            {
                "name": "zero_epochs",
                "payload": {
                    "model_tag": MINILM_MODEL_TAG,
                    "train_dataset_ids": [DATASET_ID_1],
                    "epochs": 0,  # Invalid
                    "learning_rate": 0.00005,
                    "per_device_train_batch_size": 8,
                },
                "expected_status": HTTP_422_UNPROCESSABLE_ENTITY,
            },
            {
                "name": "negative_learning_rate",
                "payload": {
                    "model_tag": MINILM_MODEL_TAG,
                    "train_dataset_ids": [DATASET_ID_1],
                    "epochs": 1,
                    "learning_rate": -0.1,  # Invalid
                    "per_device_train_batch_size": 8,
                },
                "expected_status": HTTP_422_UNPROCESSABLE_ENTITY,
            },
            {
                "name": "zero_batch_size",
                "payload": {
                    "model_tag": MINILM_MODEL_TAG,
                    "train_dataset_ids": [DATASET_ID_1],
                    "epochs": 1,
                    "learning_rate": 0.00005,
                    "per_device_train_batch_size": 0,  # Invalid
                },
                "expected_status": HTTP_422_UNPROCESSABLE_ENTITY,
            },
        ]

        for test_case in boundary_tests:
            response = client.post("/training/train", json=test_case["payload"])
            assert response.status_code == test_case["expected_status"], \
                f"Expected {test_case['expected_status']} for {test_case['name']}, " \
                f"got {response.status_code}"

    @staticmethod
    def test_training_malformed_requests(client: TestClient) -> None:
        """Test training with malformed request data."""
        malformed_requests = [
            {
                "name": "null_model_tag",
                "payload": {
                    "model_tag": None,
                    "train_dataset_ids": [DATASET_ID_1],
                    "epochs": 1,
                    "learning_rate": 0.00005,
                    "per_device_train_batch_size": 8,
                },
            },
            {
                "name": "empty_train_dataset_ids",
                "payload": {
                    "model_tag": MINILM_MODEL_TAG,
                    "train_dataset_ids": [],
                    "epochs": 1,
                    "learning_rate": 0.00005,
                    "per_device_train_batch_size": 8,
                },
            },
            {
                "name": "null_train_dataset_ids",
                "payload": {
                    "model_tag": MINILM_MODEL_TAG,
                    "train_dataset_ids": None,
                    "epochs": 1,
                    "learning_rate": 0.00005,
                    "per_device_train_batch_size": 8,
                },
            },
            {
                "name": "string_epochs",
                "payload": {
                    "model_tag": MINILM_MODEL_TAG,
                    "train_dataset_ids": [DATASET_ID_1],
                    "epochs": "not_a_number",
                    "learning_rate": 0.00005,
                    "per_device_train_batch_size": 8,
                },
            },
            {
                "name": "string_learning_rate",
                "payload": {
                    "model_tag": MINILM_MODEL_TAG,
                    "train_dataset_ids": [DATASET_ID_1],
                    "epochs": 1,
                    "learning_rate": "not_a_number",
                    "per_device_train_batch_size": 8,
                },
            },
        ]

        for test_case in malformed_requests:
            response = client.post("/training/train", json=test_case["payload"])
            assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY, \
                f"Expected 422 for {test_case['name']}, got {response.status_code}"

    @staticmethod
    def test_training_duplicate_dataset_ids(client: TestClient) -> None:
        """Test training with duplicate dataset IDs."""
        ensure_minilm_model_available()

        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "train_dataset_ids": [
                DATASET_ID_1,
                DATASET_ID_1,
                DATASET_ID_2,  # Duplicate
            ],
            "epochs": 1,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8,
        }

        response = client.post("/training/train", json=payload)
        # Should either handle duplicates gracefully or reject
        assert response.status_code in {
            HTTP_202_ACCEPTED,
            HTTP_400_BAD_REQUEST,
            HTTP_422_UNPROCESSABLE_ENTITY,
        }

        cleanup_trained_models()

    @staticmethod
    def test_training_very_long_model_tag(client: TestClient) -> None:
        """Test training with very long model tag."""
        long_model_tag = "a" * 1000  # Very long string

        payload = {
            "model_tag": long_model_tag,
            "train_dataset_ids": [DATASET_ID_1],
            "epochs": 1,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8,
        }

        response = client.post("/training/train", json=payload)
        # Should reject with 404 (model not found)
        assert response.status_code == HTTP_404_NOT_FOUND

    @staticmethod
    def test_training_special_characters_in_model_tag(client: TestClient) -> None:
        """Test training with special characters in model tag."""
        special_tags = [
            "model/with/slashes",
            "model-with-dashes",
            "model_with_underscores",
            "model.with.dots",
            "model with spaces",
            "model@with@symbols",
            "model#with#hash",
            "model$with$dollar",
            "模型中文名称",  # Chinese characters
            "🤖model-with-emoji",
        ]

        for tag in special_tags:
            payload = {
                "model_tag": tag,
                "train_dataset_ids": [DATASET_ID_1],
                "epochs": 1,
                "learning_rate": 0.00005,
                "per_device_train_batch_size": 8,
            }

            response = client.post("/training/train", json=payload)
            # Should either accept or reject appropriately - 404 for unknown models
            assert response.status_code in {
                HTTP_202_ACCEPTED,
                HTTP_404_NOT_FOUND,
                HTTP_400_BAD_REQUEST,
                HTTP_422_UNPROCESSABLE_ENTITY,
            }, (
                f"Unexpected status for tag '{tag}': {response.status_code}"
            )


@pytest.mark.edge_cases
class TestEvaluationEdgeCases:
    """Edge case tests for evaluation endpoints."""

    @staticmethod
    def test_evaluation_malformed_requests(client: TestClient) -> None:
        """Test evaluation with malformed request data."""
        malformed_requests = [
            {
                "name": "null_model_tag",
                "payload": {
                    "model_tag": None,
                    "dataset_id": DATASET_ID_1,
                },
            },
            {
                "name": "null_dataset_id",
                "payload": {
                    "model_tag": MINILM_MODEL_TAG,
                    "dataset_id": None,
                },
            },
            {
                "name": "integer_model_tag",
                "payload": {
                    "model_tag": 12345,
                    "dataset_id": DATASET_ID_1,
                },
            },
            {
                "name": "integer_dataset_id",
                "payload": {
                    "model_tag": MINILM_MODEL_TAG,
                    "dataset_id": 12345,
                },
            },
            {
                "name": "list_model_tag",
                "payload": {
                    "model_tag": ["model1", "model2"],
                    "dataset_id": DATASET_ID_1,
                },
            },
            {
                "name": "list_dataset_id",
                "payload": {
                    "model_tag": MINILM_MODEL_TAG,
                    "dataset_id": [DATASET_ID_1, DATASET_ID_2],
                },
            },
        ]

        for test_case in malformed_requests:
            response = client.post("/evaluation/evaluate", json=test_case["payload"])
            # Some cases are accepted and fail in background task
            assert response.status_code in {
                HTTP_422_UNPROCESSABLE_ENTITY,
                HTTP_202_ACCEPTED,
            }, (
                f"Expected 422 or 202 for {test_case['name']}, "
                f"got {response.status_code}"
            )

    @staticmethod
    def test_evaluation_special_characters_in_identifiers(client: TestClient) -> None:
        """Test evaluation with special characters in identifiers."""
        special_model_tags = [
            "model/with/slashes",
            "model with spaces",
            "model@with@symbols",
            "模型中文名称",  # Chinese characters
            "🤖model-with-emoji",
        ]

        for tag in special_model_tags:
            payload = {
                "model_tag": tag,
                "dataset_id": DATASET_ID_1,
            }

            response = client.post("/evaluation/evaluate", json=payload)
            # Should either accept or reject appropriately
            assert response.status_code in {
                HTTP_202_ACCEPTED,
                HTTP_400_BAD_REQUEST,
                HTTP_422_UNPROCESSABLE_ENTITY,
            }, (
                f"Unexpected status for tag '{tag}': {response.status_code}"
            )

    @staticmethod
    def test_evaluation_very_long_identifiers(client: TestClient) -> None:
        """Test evaluation with very long identifiers."""
        long_model_tag = "a" * 1000
        long_dataset_id = "b" * 1000  # Not a valid UUID format

        # Test long model tag - backend accepts this and fails in background task
        payload = {
            "model_tag": long_model_tag,
            "dataset_id": DATASET_ID_1,
        }
        response = client.post("/evaluation/evaluate", json=payload)
        assert response.status_code == HTTP_202_ACCEPTED

        # Test long dataset ID - should be rejected due to invalid UUID format
        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "dataset_id": long_dataset_id,
        }
        response = client.post("/evaluation/evaluate", json=payload)
        assert response.status_code == HTTP_404_NOT_FOUND  # Invalid UUID format


@pytest.mark.edge_cases
class TestStatusEndpointEdgeCases:
    """Edge case tests for status endpoints."""

    @staticmethod
    def test_status_malformed_task_ids(client: TestClient) -> None:
        """Test status endpoints with malformed task IDs."""
        malformed_ids = [
            "not-a-uuid",
            "12345",
            "",
            " ",
            "null",
            "undefined",
            "../../etc/passwd",  # Path traversal attempt
            "<script>alert('xss')</script>",  # XSS attempt
            "' OR 1=1 --",  # SQL injection attempt
            "a" * 1000,  # Very long string
            "模型中文名称",  # Chinese characters
            "🤖emoji-id",
        ]

        for task_id in malformed_ids:
            # Test training status
            training_response = client.get(f"/training/{task_id}/status")
            assert training_response.status_code in {
                HTTP_400_BAD_REQUEST,
                HTTP_404_NOT_FOUND,
                HTTP_422_UNPROCESSABLE_ENTITY,
            }, (
                f"Unexpected status for training ID '{task_id}': "
                f"{training_response.status_code}"
            )

            # Test evaluation status
            evaluation_response = client.get(f"/evaluation/{task_id}/status")
            assert evaluation_response.status_code in {
                HTTP_400_BAD_REQUEST,
                HTTP_404_NOT_FOUND,
                HTTP_422_UNPROCESSABLE_ENTITY,
            }, (
                f"Unexpected status for evaluation ID '{task_id}': "
                f"{evaluation_response.status_code}"
            )

    @staticmethod
    def test_status_uuid_variations(client: TestClient) -> None:
        """Test status endpoints with various UUID formats."""
        uuid_variations = [
            str(uuid.uuid4()),  # Standard UUID
            str(uuid.uuid4()).upper(),  # Uppercase
            str(uuid.uuid4()).replace("-", ""),  # No hyphens
            "00000000-0000-0000-0000-000000000000",  # All zeros
            "ffffffff-ffff-ffff-ffff-ffffffffffff",  # All f's
        ]

        for test_uuid in uuid_variations:
            # All should return 404 (not found) since they don't exist
            training_response = client.get(f"/training/{test_uuid}/status")
            evaluation_response = client.get(f"/evaluation/{test_uuid}/status")

            # Valid UUID format should get 404, invalid format should get 400/422
            if (
                len(test_uuid) == UUID_LENGTH
                and test_uuid.count("-") == UUID_HYPHEN_COUNT
            ):
                assert training_response.status_code == HTTP_404_NOT_FOUND
                assert evaluation_response.status_code == HTTP_404_NOT_FOUND
            else:
                assert training_response.status_code in {
                    HTTP_400_BAD_REQUEST,
                    HTTP_404_NOT_FOUND,
                    HTTP_422_UNPROCESSABLE_ENTITY,
                }
                assert evaluation_response.status_code in {
                    HTTP_400_BAD_REQUEST,
                    HTTP_404_NOT_FOUND,
                    HTTP_422_UNPROCESSABLE_ENTITY,
                }


@pytest.mark.edge_cases
class TestDatasetSchemaEdgeCases:
    """Edge case tests related to dataset schema handling."""

    @staticmethod
    def _load_dataset_examples(file_path: Path) -> list[dict]:
        """Load and parse examples from a dataset file."""
        examples = []
        with file_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        example = json.loads(line)
                        examples.append(example)
                    except json.JSONDecodeError:
                        pytest.fail(
                            f"Invalid JSON in {file_path.name} line {line_num}: {line}"
                        )
        return examples

    @staticmethod
    def _validate_example_schema(
        example: dict, filename: str, example_index: int
    ) -> None:
        """Validate the schema of a single dataset example."""
        required_fields = ["question", "positive", "negative"]

        for field in required_fields:
            assert field in example, (
                f"Missing '{field}' in {filename} example {example_index}"
            )
            assert isinstance(
                example[field], str
            ), f"Field '{field}' should be string in {filename} example {example_index}"
            assert example[field].strip(), (
                f"Field '{field}' should not be empty in {filename} "
                f"example {example_index}"
            )

        # Check for unexpected fields
        unexpected_fields = set(example.keys()) - set(required_fields)
        if unexpected_fields:
            # Use pytest.warns or logging if needed, but no print
            pass

    @staticmethod
    def _validate_dataset_file(filename: str) -> None:
        """Validate a single dataset file."""
        file_path = TEST_DATA_DIR / filename
        if not file_path.exists():
            pytest.skip(f"Test dataset not found: {filename}")

        examples = TestDatasetSchemaEdgeCases._load_dataset_examples(file_path)
        assert len(examples) > 0, f"No examples found in {filename}"

        # Validate schema for each example
        for i, example in enumerate(examples):
            TestDatasetSchemaEdgeCases._validate_example_schema(
                example, filename, i
            )

    @staticmethod
    def test_dataset_content_validation() -> None:
        """Test that datasets with our test schema are properly validated."""
        # Load and validate test dataset content
        test_files = [
            f"__rm_-rf__2F_{DATASET_ID_1}.jsonl",
            f"__rm_-rf__2F_{DATASET_ID_2}.jsonl",
        ]

        for filename in test_files:
            TestDatasetSchemaEdgeCases._validate_dataset_file(filename)

    @staticmethod
    def test_empty_dataset_handling(client: TestClient) -> None:
        """Test behavior with hypothetically empty datasets."""
        ensure_minilm_model_available()

        # This test assumes that empty datasets would be handled gracefully
        # We can't create empty datasets easily, but we can test with nonexistent ones
        fake_dataset_id = str(uuid.uuid4())

        # Training with nonexistent (effectively empty) dataset
        training_payload = {
            "model_tag": MINILM_MODEL_TAG,
            "train_dataset_ids": [fake_dataset_id],
            "epochs": 1,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8,
        }

        training_response = client.post("/training/train", json=training_payload)
        # Should either reject immediately or start and fail
        assert training_response.status_code in {
            HTTP_400_BAD_REQUEST,
            HTTP_404_NOT_FOUND,
            HTTP_202_ACCEPTED,
        }

        # Evaluation with nonexistent dataset
        evaluation_payload = {
            "model_tag": MINILM_MODEL_TAG,
            "dataset_id": fake_dataset_id,
        }

        evaluation_response = client.post(
            "/evaluation/evaluate", json=evaluation_payload
        )
        assert evaluation_response.status_code in {
            HTTP_400_BAD_REQUEST,
            HTTP_404_NOT_FOUND,
            HTTP_202_ACCEPTED,
        }


@pytest.mark.edge_cases
class TestConcurrencyAndStress:
    """Stress and concurrency tests."""

    @staticmethod
    def test_rapid_sequential_requests(client: TestClient) -> None:
        """Test rapid sequential training and evaluation requests."""
        ensure_minilm_model_available()

        task_ids = []

        # Send multiple requests rapidly
        for i in range(5):
            # Alternate between training and evaluation
            if i % 2 == 0:
                payload = {
                    "model_tag": MINILM_MODEL_TAG,
                    "train_dataset_ids": [DATASET_ID_1],
                    "epochs": 1,
                    "learning_rate": 0.00005,
                    "per_device_train_batch_size": 8,
                }
                response = client.post("/training/train", json=payload)
                endpoint = "training"
            else:
                payload = {
                    "model_tag": MINILM_MODEL_TAG,
                    "dataset_id": DATASET_ID_1,
                }
                response = client.post("/evaluation/evaluate", json=payload)
                endpoint = "evaluation"

            if (
                response.status_code == HTTP_202_ACCEPTED
                and response.headers.get("Location")
            ):
                match = re.search(
                    rf"/{endpoint}/([a-f0-9\-]+)/status",
                    response.headers["Location"],
                )
                if match:
                    task_ids.append((match.group(1), endpoint))

        # Check that all tasks have valid status
        for task_id, endpoint in task_ids:
            status_response = client.get(f"/{endpoint}/{task_id}/status")
            assert status_response.status_code == HTTP_200_OK
            status_data = status_response.json()
            assert status_data["status"] in {"Q", "R", "D", "F"}

        cleanup_trained_models()

    @staticmethod
    def test_request_timeout_resilience(client: TestClient) -> None:
        """Test that the API handles requests that might timeout gracefully."""
        ensure_minilm_model_available()

        # Start a training task that might take a while
        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "train_dataset_ids": [DATASET_ID_1, DATASET_ID_2],
            "epochs": 1,  # Keep low to avoid actual long runtime
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 4,
        }

        response = client.post("/training/train", json=payload)
        assert response.status_code == HTTP_202_ACCEPTED

        # The response should be immediate even if the task takes long
        assert "Location" in response.headers

        cleanup_trained_models()
