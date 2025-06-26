"""Load testing with Locust.

Run with: uvx locust -f scripts/locust.py
Run headless with: uvx locust -f scripts/locust.py --host=https://localhost --headless -u 1 -r 1
"""  # noqa: E501

import urllib3
from locust import HttpUser, constant_pacing, task

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class AppLoadTests(HttpUser):
    """Load tests for the API."""

    host = "https://localhost"
    wait_time = constant_pacing(0)

    def on_start(self) -> None:
        """Set up the client to ignore SSL certificate validation."""
        self.client.verify = False

    @task
    def get_health(self) -> None:
        """Check the health of the server."""
        self.client.get("/health")
