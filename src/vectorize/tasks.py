"""Tasks module using Dramatiq."""

import os

from dotenv import load_dotenv
from dramatiq import Broker, set_broker
from dramatiq.brokers.redis import RedisBroker
from dramatiq.brokers.stub import StubBroker
from dramatiq.middleware.asyncio import AsyncIO

load_dotenv()


def make_broker() -> Broker:
    if os.getenv("ENV") == "testing":
        broker = StubBroker()
    else:
        broker = RedisBroker(url=os.getenv("REDIS_URL", "redis://redis:6379"))
    broker.add_middleware(AsyncIO())
    return broker


set_broker(make_broker())

from vectorize.dataset.tasks import upload_hf_dataset_bg  # noqa: E402, I001

from vectorize.dataset.models import Dataset  # noqa: E402, F401
from vectorize.dataset.task_model import UploadDatasetTask  # noqa: E402, F401
from vectorize.ai_model.models import AIModel  # noqa: E402, F401
from vectorize.synthesis.models import SynthesisTask  # noqa: E402, F401
from vectorize.upload.models import UploadTask  # noqa: E402, F401

__all__ = ["upload_hf_dataset_bg"]
