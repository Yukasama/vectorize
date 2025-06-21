# ruff: noqa: S101

"""Invalid tests for actions endpoint."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient


@pytest.mark.asyncio
@pytest.mark.actions
class TestActionsInvalidParams:
    """Tests for invalid query parameter values on the /actions endpoint."""

    @classmethod
    @pytest.mark.parametrize(
        "url",
        [
            # status
            "/actions?status=PENDING",
            "/actions?status=UNKNOWN",
            "/actions?status=Q&status=FOO",
            "/actions?status=1",
            "/actions?status=",
            # completed
            "/actions?completed=123",
            "/actions?completed=yesplease",
            "/actions?completed=",
            # limit
            "/actions?limit=0",
            "/actions?limit=999",
            "/actions?limit=-5",
            "/actions?limit=abc",
            # offset
            "/actions?offset=-1",
            "/actions?offset=abc",
            # within_hours
            "/actions?within_hours=0",
            "/actions?within_hours=-1",
            "/actions?within_hours=abc",
            "/actions?within_hours=",
            # combined
            "/actions?limit=-5&offset=abc&status=WRONG&completed=1.23&within_hours=-2",
        ],
    )
    async def test_invalid_query_params(cls, client: TestClient, url: str) -> None:
        """Test that invalid query param values result in 422 Unprocessable Entity."""
        response = client.get(url)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
