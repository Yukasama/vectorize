"""Load testing with Locust."""

from pathlib import Path

from locust import HttpUser, constant_throughput, task

from tests.datasets.utils import get_test_file


class GetUser(HttpUser):
    """Load tests for Dataset requests."""

    host = "http://localhost:8000"
    wait_time = constant_throughput(0.1)

    @task(100)
    def upload_dataset(self) -> None:
        """Upload a dataset."""
        for dataset_file in [
            "default.csv",
            "default.json",
            "default.xml",
            "default.xlsx",
            "infer_fields.csv",
            "custom_fields.csv",
        ]:
            file_path = (
                Path(__file__).parent.parent
                / "test_data"
                / "datasets"
                / "valid"
                / dataset_file
            )
            files = get_test_file(file_path)
            response = self.client.post("/v1/datasets", files=files)
            print(f"{response.headers['Location'].split('/')[-1]}")  # noqa: T201
