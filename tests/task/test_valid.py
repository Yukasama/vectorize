# ruff: noqa: S101

"""Valid tests for tasks endpoint."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from vectorize.common.task_status import TaskStatus

_TASK_TYPE_OPTIONS = {"model_upload", "synthesis", "dataset_upload"}


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
    async def test_get_tasks_completed_filter(cls, client: TestClient) -> None:
        """Test tasks endpoint filtering by completed status."""
        completed_response = client.get("/tasks?completed=true")
        assert completed_response.status_code == status.HTTP_200_OK
        completed_tasks = completed_response.json()
        assert isinstance(completed_tasks, list)

        pending_response = client.get("/tasks?completed=false")
        assert pending_response.status_code == status.HTTP_200_OK
        pending_tasks = pending_response.json()
        assert isinstance(pending_tasks, list)

    @classmethod
    async def test_get_tasks_status_filter_pending(cls, client: TestClient) -> None:
        """Test tasks endpoint filtering by PENDING status."""
        response = client.get("/tasks?status=P")

        assert response.status_code == status.HTTP_200_OK
        tasks = response.json()
        assert isinstance(tasks, list)

        for action in tasks:
            assert action["task_status"] == TaskStatus.PENDING.value

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
        response = client.get("/tasks?status=P&status=D")

        assert response.status_code == status.HTTP_200_OK
        tasks = response.json()
        assert isinstance(tasks, list)

        valid_statuses = {TaskStatus.PENDING.value, TaskStatus.DONE.value}
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
        response = client.get(
            f"/tasks?limit={limit}&completed=false&status=P&within_hours=2"
        )

        assert response.status_code == status.HTTP_200_OK
        tasks = response.json()
        assert isinstance(tasks, list)
        assert len(tasks) <= limit

        for action in tasks:
            assert action["task_status"] == TaskStatus.PENDING.value
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
