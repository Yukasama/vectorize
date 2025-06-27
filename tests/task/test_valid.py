# ruff: noqa: S101

"""Valid tests for tasks endpoint."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from vectorize.common.task_status import TaskStatus

_TASK_TYPE_OPTIONS = {
    "model_upload",
    "synthesis",
    "dataset_upload",
    "training",
    "evaluation",
}


@pytest.mark.asyncio
@pytest.mark.tasks
class TestTasksValid:
    """Tests for valid tasks endpoint requests."""

    @classmethod
    async def test_get_all_tasks_default(cls, client: TestClient) -> None:
        """Test getting all tasks with default parameters."""
        default_hours = 1
        response = client.get("/tasks")

        assert response.status_code == status.HTTP_200_OK
        tasks = response.json()
        assert isinstance(tasks, list)
        assert len(tasks) >= default_hours

    @classmethod
    async def test_get_tasks_with_limit(cls, client: TestClient) -> None:
        """Test tasks endpoint with limit parameter."""
        limit = 2
        response = client.get(f"/tasks?limit={limit}")

        assert response.status_code == status.HTTP_200_OK
        tasks = response.json()
        assert isinstance(tasks, list)
        assert len(tasks) <= limit

    @classmethod
    async def test_get_tasks_with_offset(cls, client: TestClient) -> None:
        """Test tasks endpoint with offset parameter."""
        all_response = client.get("/tasks?limit=100")
        all_tasks = all_response.json()

        if len(all_tasks) > 1:
            response = client.get("/tasks?offset=1")
            assert response.status_code == status.HTTP_200_OK
            offset_tasks = response.json()
            assert isinstance(offset_tasks, list)
            assert len(offset_tasks) <= len(all_tasks)

    @classmethod
    async def test_get_tasks_status_filter_running(cls, client: TestClient) -> None:
        """Test tasks endpoint filtering by RUNNING status."""
        response = client.get("/tasks?status=R")

        assert response.status_code == status.HTTP_200_OK
        tasks = response.json()
        assert isinstance(tasks, list)

        for action in tasks:
            assert action["task_status"] == TaskStatus.RUNNING.value

    @classmethod
    async def test_get_tasks_status_filter_done(cls, client: TestClient) -> None:
        """Test tasks endpoint filtering by DONE status."""
        response = client.get("/tasks?status=D")

        assert response.status_code == status.HTTP_200_OK
        tasks = response.json()
        assert isinstance(tasks, list)

        for action in tasks:
            assert action["task_status"] == TaskStatus.DONE.value

    @classmethod
    async def test_get_tasks_multiple_status_filter(cls, client: TestClient) -> None:
        """Test tasks endpoint filtering by multiple statuses."""
        response = client.get("/tasks?status=R&status=D")

        assert response.status_code == status.HTTP_200_OK
        tasks = response.json()
        assert isinstance(tasks, list)

        valid_statuses = {TaskStatus.RUNNING.value, TaskStatus.DONE.value}
        for action in tasks:
            assert action["task_status"] in valid_statuses

    @classmethod
    async def test_get_tasks_within_hours_recent(cls, client: TestClient) -> None:
        """Test tasks endpoint with within_hours=1 (recent tasks only)."""
        response = client.get("/tasks?within_hours=1")

        assert response.status_code == status.HTTP_200_OK
        tasks = response.json()
        assert isinstance(tasks, list)

        for action in tasks:
            assert "created_at" in action
            assert "end_date" in action

    @classmethod
    async def test_get_tasks_within_hours_extended(cls, client: TestClient) -> None:
        """Test tasks endpoint with within_hours=3 (includes older tasks)."""
        results = 2
        response = client.get("/tasks?within_hours=3")

        assert response.status_code == status.HTTP_200_OK
        tasks = response.json()
        assert isinstance(tasks, list)
        assert len(tasks) >= results

    @classmethod
    async def test_get_tasks_combined_filters(cls, client: TestClient) -> None:
        """Test tasks endpoint with multiple filters combined."""
        limit = 5
        response = client.get(f"/tasks?limit={limit}&status=R&within_hours=2")

        assert response.status_code == status.HTTP_200_OK
        tasks = response.json()
        assert isinstance(tasks, list)
        assert len(tasks) <= limit

        for action in tasks:
            assert action["task_status"] == TaskStatus.RUNNING.value
            assert "created_at" in action
            assert action["task_type"] in _TASK_TYPE_OPTIONS

    @classmethod
    async def test_tasks_response_structure(cls, client: TestClient) -> None:
        """Test that tasks response has correct structure."""
        response = client.get("/tasks?limit=1")

        assert response.status_code == status.HTTP_200_OK
        tasks = response.json()
        assert isinstance(tasks, list)

        if tasks:
            action = tasks[0]
            required_fields = ["id", "task_status", "created_at", "task_type"]
            for field in required_fields:
                assert field in action

            assert action["task_type"] in _TASK_TYPE_OPTIONS
            valid_statuses = [status.value for status in TaskStatus]
            assert action["task_status"] in valid_statuses

    @classmethod
    async def test_get_tasks_tag_filter(cls, client: TestClient) -> None:
        """Test tasks endpoint filtering by specific tag."""
        response = client.get("/tasks?tag=example-hf-model")

        assert response.status_code == status.HTTP_200_OK
        tasks = response.json()
        assert isinstance(tasks, list)

        for task in tasks:
            if task["tag"] is not None:
                assert task["tag"] == "example-hf-model"

    @classmethod
    async def test_get_tasks_tag_filter_nonexistent(cls, client: TestClient) -> None:
        """Test tasks endpoint filtering by non-existent tag."""
        response = client.get("/tasks?tag=nonexistent-tag")

        assert response.status_code == status.HTTP_200_OK
        tasks = response.json()
        assert isinstance(tasks, list)
        assert len(tasks) == 0

    @classmethod
    async def test_get_tasks_single_task_type_filter(cls, client: TestClient) -> None:
        """Test tasks endpoint filtering by single task type."""
        response = client.get("/tasks?task_type=model_upload")

        assert response.status_code == status.HTTP_200_OK
        tasks = response.json()
        assert isinstance(tasks, list)

        for task in tasks:
            assert task["task_type"] == "model_upload"

    @classmethod
    async def test_get_tasks_multiple_task_type_filter(cls, client: TestClient) -> None:
        """Test tasks endpoint filtering by multiple task types."""
        response = client.get("/tasks?task_type=model_upload&task_type=dataset_upload")

        assert response.status_code == status.HTTP_200_OK
        tasks = response.json()
        assert isinstance(tasks, list)

        valid_types = {"model_upload", "dataset_upload"}
        for task in tasks:
            assert task["task_type"] in valid_types

    @classmethod
    async def test_get_tasks_task_type_filter_all_types(
        cls, client: TestClient
    ) -> None:
        """Test tasks endpoint with all possible task types."""
        all_types = [
            "model_upload",
            "synthesis",
            "dataset_upload",
            "training",
            "evaluation",
        ]

        for task_type in all_types:
            response = client.get(f"/tasks?task_type={task_type}")

            assert response.status_code == status.HTTP_200_OK
            tasks = response.json()
            assert isinstance(tasks, list)

            for task in tasks:
                assert task["task_type"] == task_type

    @classmethod
    async def test_get_tasks_combined_tag_and_task_type_filter(
        cls, client: TestClient
    ) -> None:
        """Test tasks endpoint with both tag and task type filters."""
        response = client.get("/tasks?tag=example-hf-model&task_type=model_upload")

        assert response.status_code == status.HTTP_200_OK
        tasks = response.json()
        assert isinstance(tasks, list)

        for task in tasks:
            assert task["task_type"] == "model_upload"
            if task["tag"] is not None:
                assert task["tag"] == "example-hf-model"

    @classmethod
    async def test_get_tasks_complex_filters_combination(
        cls, client: TestClient
    ) -> None:
        """Test tasks endpoint with tag, task type, status, and limit filters."""
        limit = 5
        response = client.get(
            f"/tasks?tag=training_task&task_type=training&status=R&limit={limit}"
        )

        assert response.status_code == status.HTTP_200_OK
        tasks = response.json()
        assert isinstance(tasks, list)
        assert len(tasks) <= limit

        for task in tasks:
            assert task["task_type"] == "training"
            assert task["task_status"] == TaskStatus.RUNNING.value
            if task["tag"] is not None:
                assert task["tag"] == "training_task"

    @classmethod
    async def test_get_tasks_no_task_type_returns_all(cls, client: TestClient) -> None:
        """Test that not specifying task_type returns all task types."""
        response = client.get("/tasks?limit=100")

        assert response.status_code == status.HTTP_200_OK
        tasks = response.json()
        assert isinstance(tasks, list)

        found_types = {task["task_type"] for task in tasks}
        assert len(found_types) > 1
        assert found_types.issubset(_TASK_TYPE_OPTIONS)

    @classmethod
    async def test_get_tasks_empty_task_type_array(cls, client: TestClient) -> None:
        """Test tasks endpoint behavior with empty task type array."""
        response = client.get("/tasks?task_type=")

        assert response.status_code in {
            status.HTTP_200_OK,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        }

    @classmethod
    async def test_get_tasks_invalid_task_type(cls, client: TestClient) -> None:
        """Test tasks endpoint with invalid task type."""
        response = client.get("/tasks?task_type=invalid_type")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
