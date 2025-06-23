# ruff: noqa: S101

"""Invalid tests for tasks endpoint."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient


@pytest.mark.asyncio
@pytest.mark.tasks
class TestActionsInvalidParams:
    """Tests for invalid query parameter values on the /tasks endpoint."""

    @classmethod
    @pytest.mark.parametrize(
        "url",
        [
            # status
            "/tasks?status=PENDING",
            "/tasks?status=UNKNOWN",
            "/tasks?status=Q&status=FOO",
            "/tasks?status=1",
            "/tasks?status=",
            # completed
            "/tasks?completed=123",
            "/tasks?completed=yesplease",
            "/tasks?completed=",
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
            # combined
            "/tasks?limit=-5&offset=abc&status=WRONG&completed=1.23&within_hours=-2",
        ],
    )
    async def test_invalid_query_params(cls, client: TestClient, url: str) -> None:
        """Test that invalid query param values result in 422 Unprocessable Entity."""
        response = client.get(url)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
