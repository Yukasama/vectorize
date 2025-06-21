"""Evaluation router."""
from pathlib import Path
from typing import Annotated

import torch
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from loguru import logger
from mteb import MTEB
from sentence_transformers import SentenceTransformer
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.ai_model.exceptions import ModelNotFoundError
from vectorize.ai_model.models import AIModel
from vectorize.ai_model.repository import get_ai_model_db
from vectorize.config.db import get_session

__all__ = ["router"]

router = APIRouter(tags=["Evaluation"])
CACHE_BASE_DIR = "/"
SELECTED_MTEB_TASKS = ["STSBenchmark", "BIOSSES", "SICK-R"]


# - Get CACHE_BASE_DIR configurable from env or setting and move to utils
def resolve_cache_path(base_cache_dir: str, model: AIModel) -> str:
    """Temporarily resolve the cache path for a given AIModel."""
    source_map = {
        "github": "gh_cache",
        "huggingface": "hf_cache",
        "local": "local_cache",
    }

    source_dir = source_map.get(str(model.source))
    if not source_dir:
        raise ValueError(f"Unsupported model source: {model.source}")

    return str(Path(base_cache_dir) / source_dir / model.model_tag)


# - Add to BG Tasks and add service
@router.get("/mteb/{model_tag}", summary="Run benchmark on a cached model by UUID")
async def get_evaluation_results(
    model_tag: str,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> JSONResponse:
    """Run MTEB evaluation tasks on a cached model by its UUID.

    Args:
        model_tag (str): The unique identifier for the AIModel.
        db (AsyncSession): The database session dependency.

    Returns:
        JSONResponse: The results of the MTEB evaluation tasks.

    Raises:
        HTTPException: If the model is not found or if the benchmark execution fails.
    """
    logger.debug("Running MTEB evaluation tasks {} for model_tag: {}",
        SELECTED_MTEB_TASKS,
        model_tag)

    model = await get_ai_model_db(db, model_tag)
    if not model:
        logger.error("Model with tag {} not found", model_tag)
        raise ModelNotFoundError()

    logger.debug("Retrieved model: {}", model)
    use_cuda = torch.cuda.is_available()
    logger.debug("CUDA available: {}", use_cuda)
    device_name = torch.cuda.get_device_name(0) if use_cuda else "No GPU"
    logger.debug("Device: {}", device_name)

    model_cached_path = resolve_cache_path(CACHE_BASE_DIR, model)
    logger.debug("Resolved cache path for model {}: {}",
        model_tag,
        model_cached_path)
    try:
        model = SentenceTransformer(model_cached_path)
        if use_cuda:
            model = model.to("cuda")  # Force GPU usage
            device = next(model.parameters()).device
            logger.debug("Model loaded to device: {}", device)
        else:
            logger.warning("CUDA not available. Model will run on CPU.")
        benchmark = MTEB(tasks=SELECTED_MTEB_TASKS)
        results = benchmark.run(model=model)

        logger.debug("Benchmark completed with results: {}", results)
    except Exception as e:
        logger.error("Benchmark execution failed: {}", e)
        return JSONResponse(
            content={
                "error": "Benchmark execution failed",
                "details": str(e)}, status_code=500)

    return JSONResponse(content=results)
