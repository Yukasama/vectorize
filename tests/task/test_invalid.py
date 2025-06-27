# ruff: noqa: S101

"""Invalid tests for tasks endpoint."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient


@pytest.mark.asyncio
@pytest.mark.tasks
class TestTasksInvalidParams:
    """Tests for invalid query parameter values on the /tasks endpoint."""

    @classmethod
    @pytest.mark.parametrize(
        "url",
        [
            # status
            "/tasks?status=RUNNING",
            "/tasks?status=UNKNOWN",
            "/tasks?status=Q&status=FOO",
            "/tasks?status=1",
            "/tasks?status=",
            # limit
            "/tasks?limit=0",
            "/tasks?limit=999",
            "/tasks?limit=-5",
            "/tasks?limit=abc",
            # offset
            "/tasks?offset=-1",
            "/tasks?offset=abc",
            # within_hours
            "/tasks?within_hours=0",
            "/tasks?within_hours=-1",
            "/tasks?within_hours=abc",
            "/tasks?within_hours=",
            # task_type
            "/tasks?task_type=invalid_type",
            "/tasks?task_type=UPLOAD",
            "/tasks?task_type=model_upload_wrong",
            "/tasks?task_type=123",
            "/tasks?task_type=training&task_type=invalid_type",
            "/tasks?task_type=",
            # tag
            "/tasks?tag=/tasks?tag=" + "x" * 256,
            # combined
            "/tasks?limit=-5&offset=abc&status=WRONG&completed=1.23&within_hours=-2",
            "/tasks?task_type=invalid&tag=&limit=abc",
            "/tasks?task_type=wrong_type&tag=invalid<>tag&status=UNKNOWN",
        ],
    )
    async def test_invalid_query_params(cls, client: TestClient, url: str) -> None:
        """Test that invalid query param values result in 422 Unprocessable Entity."""
        response = client.get(url)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
