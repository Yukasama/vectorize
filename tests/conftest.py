"""Common test fixtures for the application."""

import os
from pathlib import Path

import redis

os.environ.setdefault("ENV", "testing")

import signal
import subprocess  # noqa: S404
import sys
import time
from collections.abc import AsyncGenerator, Generator

import pytest
from fastapi.testclient import TestClient
from loguru import logger
from redis import Redis
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from testcontainers.redis import RedisContainer

from vectorize.app import app
from vectorize.config import settings
from vectorize.config.db import get_session
from vectorize.config.seed import seed_db

# Import aller SQLModel-Tabellen, damit sie beim create_all verfügbar sind
from vectorize.evaluation.models import EvaluationTask
from vectorize.training.models import TrainingTask
from vectorize.upload.models import UploadTask
from vectorize.synthesis.models import SynthesisTask
from vectorize.ai_model.models import AIModel
from vectorize.inference.models import InferenceCounter
from vectorize.dataset.task_model import UploadDatasetTask
from vectorize.dataset.models import Dataset

# Explicitly ensure all models are loaded by referencing them
_MODELS = [EvaluationTask, TrainingTask, UploadTask, SynthesisTask, AIModel, InferenceCounter, UploadDatasetTask, Dataset]

REDIS_TEST_PORT = 56379


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_db() -> Generator[None]:
    """Clean up test database before and after each test."""
    db_path = Path("app.db")
    if db_path.exists():
        db_path.unlink()
    yield
    if db_path.exists():
        db_path.unlink()


@pytest.fixture(autouse=True)
def flush_redis() -> None:
    """Flush Redis before each test to ensure a clean state."""
    redis_url = f"redis://localhost:{REDIS_TEST_PORT}/0"
    os.environ["REDIS_URL"] = redis_url
    r = redis.from_url(redis_url)
    r.flushall()
    r.script_flush()


@pytest.fixture(scope="session", autouse=True)
def redis_container() -> Generator[RedisContainer]:
    """Fixture to start a Redis container for testing."""
    container: RedisContainer = (
        RedisContainer("redis:7.2-alpine")
        .with_bind_ports(6379, REDIS_TEST_PORT)
        .with_env("REDIS_REPLICATION_MODE", "master")
        .with_env("save", "")
        .with_env("appendonly", "no")
        .start()
    )

    redis_url = f"redis://localhost:{REDIS_TEST_PORT}/0"
    os.environ["REDIS_URL"] = redis_url
    Redis.from_url(redis_url).ping()

    yield container
    container.stop()


@pytest.fixture(scope="session", autouse=True)
def dramatiq_worker(redis_container: RedisContainer) -> Generator[None]:  # noqa: ARG001
    """Fixture to start a Dramatiq worker for testing."""
    cmd = [sys.executable, "-m", "dramatiq", "vectorize.tasks", "-p", "1", "-t", "4"]
    worker = subprocess.Popen(cmd, env=os.environ.copy())  # noqa: S603
    time.sleep(2)

    if worker.poll() is not None:
        raise RuntimeError("Dramatiq worker crashed")

    yield
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            worker.send_signal(sig)
            worker.wait(timeout=10)
            break
        except Exception:
            logger.info("Dramatiq worker couldn't be stopped gracefully", exc_info=True)
    else:
        worker.kill()


@pytest.fixture(scope="session")
async def session(cleanup_test_db: Generator[None]) -> AsyncGenerator[AsyncSession]:  # noqa: ARG001
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
        
        # Additional safety check: verify table exists in database
        result = await conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='evaluation_task';")
        )
        tables_in_db = result.fetchall()
        print(f"evaluation_task in database: {len(tables_in_db) > 0}")
        if len(tables_in_db) == 0:
            raise RuntimeError("evaluation_task table was not created in database!")

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
