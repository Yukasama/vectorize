# ruff: noqa: S101

"""Valid tests for actions endpoint."""

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
@pytest.mark.actions
class TestActionsValid:
    """Tests for valid actions endpoint requests."""

    @classmethod
    async def test_get_all_actions_default(cls, client: TestClient) -> None:
        """Test getting all actions with default parameters."""
        default_hours = 1
        response = client.get("/actions")

        assert response.status_code == status.HTTP_200_OK
        actions = response.json()
        assert isinstance(actions, list)
        assert len(actions) >= default_hours

    @classmethod
    async def test_get_actions_with_limit(cls, client: TestClient) -> None:
        """Test actions endpoint with limit parameter."""
        limit = 2
        response = client.get(f"/actions?limit={limit}")

        assert response.status_code == status.HTTP_200_OK
        actions = response.json()
        assert isinstance(actions, list)
        assert len(actions) <= limit

    @classmethod
    async def test_get_actions_with_offset(cls, client: TestClient) -> None:
        """Test actions endpoint with offset parameter."""
        all_response = client.get("/actions?limit=100")
        all_actions = all_response.json()

        if len(all_actions) > 1:
            response = client.get("/actions?offset=1")
            assert response.status_code == status.HTTP_200_OK
            offset_actions = response.json()
            assert isinstance(offset_actions, list)
            assert len(offset_actions) <= len(all_actions)

    @classmethod
    async def test_get_actions_completed_filter(cls, client: TestClient) -> None:
        """Test actions endpoint filtering by completed status."""
        completed_response = client.get("/actions?completed=true")
        assert completed_response.status_code == status.HTTP_200_OK
        completed_actions = completed_response.json()
        assert isinstance(completed_actions, list)

        pending_response = client.get("/actions?completed=false")
        assert pending_response.status_code == status.HTTP_200_OK
        pending_actions = pending_response.json()
        assert isinstance(pending_actions, list)

    @classmethod
    async def test_get_actions_status_filter_pending(cls, client: TestClient) -> None:
        """Test actions endpoint filtering by PENDING status."""
        response = client.get("/actions?status=P")

        assert response.status_code == status.HTTP_200_OK
        actions = response.json()
        assert isinstance(actions, list)

        for action in actions:
            assert action["task_status"] == TaskStatus.PENDING.value

    @classmethod
    async def test_get_actions_status_filter_done(cls, client: TestClient) -> None:
        """Test actions endpoint filtering by DONE status."""
        response = client.get("/actions?status=D")

        assert response.status_code == status.HTTP_200_OK
        actions = response.json()
        assert isinstance(actions, list)

        for action in actions:
            assert action["task_status"] == TaskStatus.DONE.value

    @classmethod
    async def test_get_actions_multiple_status_filter(cls, client: TestClient) -> None:
        """Test actions endpoint filtering by multiple statuses."""
        response = client.get("/actions?status=P&status=D")

        assert response.status_code == status.HTTP_200_OK
        actions = response.json()
        assert isinstance(actions, list)

        valid_statuses = {TaskStatus.PENDING.value, TaskStatus.DONE.value}
        for action in actions:
            assert action["task_status"] in valid_statuses

    @classmethod
    async def test_get_actions_within_hours_recent(cls, client: TestClient) -> None:
        """Test actions endpoint with within_hours=1 (recent tasks only)."""
        response = client.get("/actions?within_hours=1")

        assert response.status_code == status.HTTP_200_OK
        actions = response.json()
        assert isinstance(actions, list)

        for action in actions:
            assert "created_at" in action
            assert "end_date" in action

    @classmethod
    async def test_get_actions_within_hours_extended(cls, client: TestClient) -> None:
        """Test actions endpoint with within_hours=3 (includes older tasks)."""
        results = 2
        response = client.get("/actions?within_hours=3")

        assert response.status_code == status.HTTP_200_OK
        actions = response.json()
        assert isinstance(actions, list)
        assert len(actions) >= results

    @classmethod
    async def test_get_actions_combined_filters(cls, client: TestClient) -> None:
        """Test actions endpoint with multiple filters combined."""
        limit = 5
        response = client.get(
            f"/actions?limit={limit}&completed=false&status=P&within_hours=2"
        )

        assert response.status_code == status.HTTP_200_OK
        actions = response.json()
        assert isinstance(actions, list)
        assert len(actions) <= limit

        for action in actions:
            assert action["task_status"] == TaskStatus.PENDING.value
            assert "created_at" in action
            assert action["task_type"] in _TASK_TYPE_OPTIONS

    @classmethod
    async def test_actions_response_structure(cls, client: TestClient) -> None:
        """Test that actions response has correct structure."""
        response = client.get("/actions?limit=1")

        assert response.status_code == status.HTTP_200_OK
        actions = response.json()
        assert isinstance(actions, list)

        if actions:
            action = actions[0]
            required_fields = ["id", "task_status", "created_at", "task_type"]
            for field in required_fields:
                assert field in action

            assert action["task_type"] in _TASK_TYPE_OPTIONS
            valid_statuses = [status.value for status in TaskStatus]
            assert action["task_status"] in valid_statuses
