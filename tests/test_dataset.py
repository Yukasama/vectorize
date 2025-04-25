# ruff: noqa: S101

"""Dataset tests."""

import asyncio
import os
from pathlib import Path
from uuid import UUID

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, StaticPool
from sqlmodel.ext.asyncio.session import AsyncSession

from txt2vec.app import app
from txt2vec.config.config import prefix
from txt2vec.config.db import get_session
from txt2vec.datasets.repository import get_dataset

TRAINING_FOLDER = "test_data"
TEST_FILE_NAME = "trainingdata"
INVALID_FORMAT_NAME = "trainingdata_wrong_format.csv"
EMPTY_FILE_NAME = "trainingdata_empty.csv"
UNSUPPORTED_FORMAT = "trainingdata_unsupported_format.txt"


@pytest.fixture(scope="session")
async def session():
    """Create a test database engine."""
    test_engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )

    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async with AsyncSession(test_engine) as session:
        yield session


@pytest.fixture(name="client")
def client_fixture(session: AsyncSession):
    """Create a test client for the FastAPI app."""

    def get_session_override():
        return session

    app.dependency_overrides[get_session] = get_session_override

    client = TestClient(app)

    yield client

    app.dependency_overrides.clear()


@pytest.mark.asyncio
@pytest.mark.dataset
@pytest.mark.parametrize(
    "file_name,mime_type",
    [
        (f"{TEST_FILE_NAME}.csv", "text/csv"),
        (f"{TEST_FILE_NAME}.json", "application/json"),
        (f"{TEST_FILE_NAME}.xml", "application/xml"),
    ],
)
async def test_dataset_formats_upload(
    file_name: str, mime_type: str, session: AsyncSession, client: TestClient
) -> None:
    """Parametrized test for uploading multiple file formats."""
    base_dir = Path(__file__).parent.parent / TRAINING_FOLDER / "datasets"
    test_file_path = base_dir / file_name

    file_content = Path(test_file_path).read_bytes()
    files = {"file": (os.path.basename(test_file_path), file_content, mime_type)}

    response = client.post(f"{prefix}/datasets", files=files)
    assert response.status_code == status.HTTP_201_CREATED

    dataset_id = response.headers["Location"].split("/")[-1]
    assert dataset_id is not None

    max_retries = 3
    for attempt in range(max_retries):
        try:
            dataset = await get_dataset(db=session, dataset_id=UUID(dataset_id))
            assert dataset.id == UUID(dataset_id)
            break
        except Exception:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(0.5)


@pytest.mark.dataset
def test_dataset_invalid_format(client: TestClient) -> None:
    """Test uploading an invalid file format."""
    base_dir = Path(__file__).parent.parent / TRAINING_FOLDER / "datasets"
    test_file_path = base_dir / INVALID_FORMAT_NAME

    file_content = Path(test_file_path).read_bytes()
    files = {"file": (os.path.basename(test_file_path), file_content, "text/csv")}

    response = client.post(f"{prefix}/datasets", files=files)

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["code"] == "INVALID_CSV_FORMAT"


@pytest.mark.dataset
def test_dataset_empty(client: TestClient) -> None:
    """Test uploading an empty file."""
    base_dir = Path(__file__).parent.parent / TRAINING_FOLDER / "datasets"
    test_file_path = base_dir / EMPTY_FILE_NAME

    file_content = Path(test_file_path).read_bytes()
    files = {"file": (os.path.basename(test_file_path), file_content, "text/csv")}

    response = client.post(f"{prefix}/datasets", files=files)

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["code"] == "EMPTY_FILE"


@pytest.mark.dataset
def test_dataset_unsupported_format(client: TestClient) -> None:
    """Test uploading an unsupported format."""
    base_dir = Path(__file__).parent.parent / TRAINING_FOLDER / "datasets"
    test_file_path = base_dir / UNSUPPORTED_FORMAT

    file_content = Path(test_file_path).read_bytes()
    files = {"file": (os.path.basename(test_file_path), file_content, "text/csv")}

    response = client.post(f"{prefix}/datasets", files=files)

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["code"] == "UNSUPPORTED_FORMAT"
