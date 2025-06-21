"""Evaluation router."""
import json
import traceback
from pathlib import Path
from typing import Annotated

import torch
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from loguru import logger
from mteb import MTEB, get_tasks
from sentence_transformers import SentenceTransformer
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.ai_model.exceptions import ModelNotFoundError
from vectorize.ai_model.model_source import ModelSource
from vectorize.ai_model.models import AIModel
from vectorize.ai_model.repository import get_ai_model_db
from vectorize.config.db import get_session

__all__ = ["router"]

router = APIRouter(tags=["Evaluation"])
CACHE_BASE_DIR = "/app/data/models"
SELECTED_MTEB_TASKS = ["STSBenchmark", "BIOSSES", "SICK-R"]


def resolve_cache_path(model: AIModel) -> Path:
    """Resolve the full local snapshot path for a cached model."""
    source_map = {
        ModelSource.HUGGINGFACE: "hf_cache",
        ModelSource.GITHUB: "gh_cache",
        ModelSource.LOCAL: "local_cache",
    }

    source_dir = source_map.get(model.source)
    if not source_dir:
        raise ValueError(f"Unsupported model source: {model.source}")
# ruff: noqa: E501
    # Hugging Face-style cache structure
    base_path = Path(CACHE_BASE_DIR) / f"models--{model.name.replace('/', '--')}"
    snapshot_dir = base_path / "snapshots"

    if not snapshot_dir.exists():
        raise FileNotFoundError(f"Snapshot folder not found at: {snapshot_dir}")
# ruff: noqa: E501
    snapshots = [p for p in snapshot_dir.iterdir() if p.is_dir()]
    if not snapshots:
        raise FileNotFoundError(f"No snapshot folders found in: {snapshot_dir}")
    if len(snapshots) > 1:
        raise RuntimeError(f"Multiple snapshot folders found in: {snapshot_dir}. Ambiguous resolution.")

    return snapshots[0]


# - Add to BG Tasks and add service
@router.get("/mteb/{model_tag}", summary="Run benchmark on a cached model by UUID")
async def get_evaluation_results(
    model_tag: str,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> JSONResponse:
    logger.debug("Running MTEB evaluation tasks {} for model_tag: {}", SELECTED_MTEB_TASKS, model_tag)

    model = await get_ai_model_db(db, model_tag)
    if not model:
        logger.error("Model with tag {} not found", model_tag)
        raise ModelNotFoundError()

    logger.debug("Retrieved model: {}", model)
    use_cuda = torch.cuda.is_available()
    logger.debug("CUDA available: {}", use_cuda)
    device_name = torch.cuda.get_device_name(0) if use_cuda else "No GPU"
    logger.debug("Device: {}", device_name)

    model_cached_path = resolve_cache_path(model)
    logger.debug("Resolved cache path for model {}: {}", model_tag, model_cached_path)

    try:
        tasks = get_tasks(tasks=SELECTED_MTEB_TASKS)
        model = SentenceTransformer(str(model_cached_path))
        if use_cuda:
            model = model.to("cuda")
            device = next(model.parameters()).device
            logger.debug("Model loaded to device: {}", device)
        else:
            logger.warning("CUDA not available. Benchmark will run on CPU.")

        # Run benchmark
        benchmark = MTEB(tasks=tasks)
        results = benchmark.run(model=model, output_folder=None, eval_splits=["test"], eval_subsets=[])

        # Serialization
        def serialize_task_result(obj):
            if obj.__class__.__name__ == "TaskResult":
                return obj.__dict__
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        results_json = json.loads(json.dumps(results, default=serialize_task_result))
        return JSONResponse(content=results_json)

    except Exception as e:
        logger.error("Benchmark execution failed: {}\n{}", e, traceback.format_exc())
        return JSONResponse(
            content={
                "error": "Benchmark execution failed",
                "details": str(e),
                "stack_trace": traceback.format_exc()
            },
            status_code=500
        )
