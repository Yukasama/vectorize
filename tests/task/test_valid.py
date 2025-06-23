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
}


@pytest.mark.asyncio
@pytest.mark.tasks
class TestTasksValid:
    """Tests for valid tasks endpoint requests."""

    @classmethod
    async def test_get_all_actions_default(cls, client: TestClient) -> None:
        """Test getting all tasks with default parameters."""
        default_hours = 1
        response = client.get("/tasks")

        assert response.status_code == status.HTTP_200_OK
        tasks = response.json()
        assert isinstance(tasks, list)
        assert len(tasks) >= default_hours

    @classmethod
    async def test_get_actions_with_limit(cls, client: TestClient) -> None:
        """Test tasks endpoint with limit parameter."""
        limit = 2
        response = client.get(f"/tasks?limit={limit}")

        assert response.status_code == status.HTTP_200_OK
        tasks = response.json()
        assert isinstance(tasks, list)
        assert len(tasks) <= limit

    @classmethod
    async def test_get_actions_with_offset(cls, client: TestClient) -> None:
        """Test tasks endpoint with offset parameter."""
        all_response = client.get("/tasks?limit=100")
        all_actions = all_response.json()

        if len(all_actions) > 1:
            response = client.get("/tasks?offset=1")
            assert response.status_code == status.HTTP_200_OK
            offset_actions = response.json()
            assert isinstance(offset_actions, list)
            assert len(offset_actions) <= len(all_actions)

    @classmethod
    async def test_get_actions_completed_filter(cls, client: TestClient) -> None:
        """Test tasks endpoint filtering by completed status."""
        completed_response = client.get("/tasks?completed=true")
        assert completed_response.status_code == status.HTTP_200_OK
        completed_actions = completed_response.json()
        assert isinstance(completed_actions, list)

        pending_response = client.get("/tasks?completed=false")
        assert pending_response.status_code == status.HTTP_200_OK
        pending_actions = pending_response.json()
        assert isinstance(pending_actions, list)

    @classmethod
    async def test_get_actions_status_filter_pending(cls, client: TestClient) -> None:
        """Test tasks endpoint filtering by PENDING status."""
        response = client.get("/tasks?status=P")

        assert response.status_code == status.HTTP_200_OK
        tasks = response.json()
        assert isinstance(tasks, list)

        for action in tasks:
            assert action["task_status"] == TaskStatus.PENDING.value

    @classmethod
    async def test_get_actions_status_filter_done(cls, client: TestClient) -> None:
        """Test tasks endpoint filtering by DONE status."""
        response = client.get("/tasks?status=D")

        assert response.status_code == status.HTTP_200_OK
        tasks = response.json()
        assert isinstance(tasks, list)

        for action in tasks:
            assert action["task_status"] == TaskStatus.DONE.value

    @classmethod
    async def test_get_actions_multiple_status_filter(cls, client: TestClient) -> None:
        """Test tasks endpoint filtering by multiple statuses."""
        response = client.get("/tasks?status=P&status=D")

        assert response.status_code == status.HTTP_200_OK
        tasks = response.json()
        assert isinstance(tasks, list)

        valid_statuses = {TaskStatus.PENDING.value, TaskStatus.DONE.value}
        for action in tasks:
            assert action["task_status"] in valid_statuses

    @classmethod
    async def test_get_actions_within_hours_recent(cls, client: TestClient) -> None:
        """Test tasks endpoint with within_hours=1 (recent tasks only)."""
        response = client.get("/tasks?within_hours=1")

        assert response.status_code == status.HTTP_200_OK
        tasks = response.json()
        assert isinstance(tasks, list)

        for action in tasks:
            assert "created_at" in action
            assert "end_date" in action

    @classmethod
    async def test_get_actions_within_hours_extended(cls, client: TestClient) -> None:
        """Test tasks endpoint with within_hours=3 (includes older tasks)."""
        results = 2
        response = client.get("/tasks?within_hours=3")

        assert response.status_code == status.HTTP_200_OK
        tasks = response.json()
        assert isinstance(tasks, list)
        assert len(tasks) >= results

    @classmethod
    async def test_get_actions_combined_filters(cls, client: TestClient) -> None:
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
    async def test_actions_response_structure(cls, client: TestClient) -> None:
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
