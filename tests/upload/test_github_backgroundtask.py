"""Tests for the GitHub backgroundtask."""


import pytest


@pytest.mark.asyncio
async def test_task_pending() -> None:
    """Test for setting up PENDING TASK."""


@pytest.mark.asyncio
async def test_task_completed() -> None:
    """Test for setting up COMPLETED TASK."""


@pytest.mark.asyncio
async def test_task_cancelled() -> None:
    """Test for setting up CANCELLED TASK."""


@pytest.mark.asyncio
async def test_task_finished() -> None:
    """Test for setting up FINISHED TASK."""
