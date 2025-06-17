"""Common test fixtures for the application."""

import os
import glob
import shutil
import warnings
from collections.abc import AsyncGenerator, Generator

os.environ.setdefault("ENV", "testing")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, StaticPool
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.app import app
from vectorize.config import settings
from vectorize.config.db import get_session
from vectorize.config.seed import seed_db


def pytest_configure(config: object) -> None:
    """Configure pytest to filter warnings for MPS pin_memory on Mac."""
    del config  # unused
    warnings.filterwarnings(
        "ignore",
        message=".*pin_memory.*not supported on MPS.*",
        category=UserWarning,
    )


@pytest.fixture(scope="session")
async def session() -> AsyncGenerator[AsyncSession]:
    """Create a test database engine.

    Returns:
        AsyncSession: SQLModel async session for database operations.
    """
    test_engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False, "timeout": 30},
        poolclass=StaticPool,
        echo=False,
    )

    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async with AsyncSession(test_engine) as session:
        await seed_db(session)

    async with AsyncSession(test_engine) as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@pytest.fixture(name="client")
def client_fixture(session: AsyncSession) -> Generator:
    """Create a test client for the FastAPI app.

    Args:
        session: Database session fixture.

    Returns:
        TestClient: Configured FastAPI test client.
    """

    def get_session_override() -> AsyncSession:
        return session

    app.dependency_overrides[get_session] = get_session_override

    client = TestClient(app, base_url=f"http://testserver{settings.prefix}")  # NOSONAR
    yield client

    app.dependency_overrides.clear()


def cleanup_temporary_test_files() -> None:
    """Clean up temporary dataset and model files created during tests."""
    if not settings.app_env == "testing":
        return
    
    # Clean up temporary dataset files (those with UUID patterns in the name)
    dataset_pattern = os.path.join(settings.dataset_upload_dir, "*-*-*-*-*.jsonl")
    for file_path in glob.glob(dataset_pattern):
        try:
            os.remove(file_path)
        except OSError:
            pass  # File might already be deleted
    
    # Clean up temporary model directories
    model_pattern = os.path.join(settings.model_upload_dir, "trained_models", "*")
    for dir_path in glob.glob(model_pattern):
        try:
            if os.path.isdir(dir_path):
                shutil.rmtree(dir_path)
        except OSError:
            pass  # Directory might already be deleted


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Automatically clean up temporary files after each test."""
    yield  # Run the test
    cleanup_temporary_test_files()  # Clean up after the test


def pytest_sessionfinish(session, exitstatus):
    """Clean up temporary files after all tests are done."""
    del session, exitstatus  # unused
    cleanup_temporary_test_files()
