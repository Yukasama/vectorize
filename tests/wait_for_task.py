"""Utility to wait for task completion in FastAPI tests."""

import asyncio
import time
from typing import Any

from fastapi import status
from fastapi.testclient import TestClient

from vectorize.common.task_status import TaskStatus

__all__ = ["wait_for_task"]


_DEFAULT_TIMEOUT = 45  # seconds


async def wait_for_task(
    client: TestClient,
    task_id: str,
    status_path: str,
    poll_interval: float = 1.0,
    timeout_value: int = _DEFAULT_TIMEOUT,
) -> dict[str, Any] | None:
    """Poll task status until completion or timeout.

    Args:
        client: FastAPI test client
        task_id: ID of the task to monitor
        status_path: Status endpoint (e.g., "/datasets/huggingface/status/{}")
        poll_interval: Time between status checks in seconds
        timeout_value: Maximum time to wait in seconds

    Returns:
        Task data dictionary if completed, None if failed

    Raises:
        TimeoutError: If task doesn't complete within timeout
    """
    start_time = time.time()

    while time.time() - start_time < timeout_value:
        status_response = client.get(status_path.format(task_id))

        if status_response.status_code == status.HTTP_200_OK:
            data = status_response.json()
            if data["task_status"] in {TaskStatus.DONE, TaskStatus.FAILED}:
                return data

        await asyncio.sleep(poll_interval)

    raise TimeoutError(
        f"Task {task_id} did not complete within {timeout_value} seconds"
    )
