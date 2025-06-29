"""Tasks for dramatiq."""

import os

import dramatiq
from dotenv import load_dotenv
from dramatiq.brokers.redis import RedisBroker
from dramatiq.middleware.asyncio import AsyncIO

load_dotenv()


url = os.environ["REDIS_URL"]
broker = RedisBroker(url=url)
broker.add_middleware(AsyncIO())
dramatiq.set_broker(broker)

from vectorize.dataset.tasks import upload_hf_dataset_bg  # noqa: E402, I001
from vectorize.evaluation.tasks import run_evaluation_bg  # noqa: E402
from vectorize.training.tasks import run_training_bg  # noqa: E402

from vectorize.ai_model.models import AIModel  # noqa: E402, F401
from vectorize.dataset.models import Dataset  # noqa: E402, F401
from vectorize.dataset.task_model import UploadDatasetTask  # noqa: E402, F401
from vectorize.synthesis.models import SynthesisTask  # noqa: E402, F401
from vectorize.upload.models import UploadTask  # noqa: E402, F401
from vectorize.training.models import TrainingTask  # noqa: E402, F401
from vectorize.evaluation.models import EvaluationTask  # noqa: E402, F401

__all__ = ["run_evaluation_bg", "run_training_bg", "upload_hf_dataset_bg"]
