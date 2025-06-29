"""Router for model evaluation endpoints."""

from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Response, status
from fastapi.responses import JSONResponse
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.config.db import get_session
from vectorize.training.exceptions import InvalidDatasetIdError

from .exceptions import EvaluationTaskNotFoundError
from .models import EvaluationTask
from .repository import get_evaluation_task_by_id_db, save_evaluation_task_db
from .schemas import EvaluationRequest, EvaluationStatusResponse
from .service import EvaluationService
from .tasks import run_evaluation_bg

__all__ = ["router"]

router = APIRouter(tags=["Evaluation"])


@router.post("/evaluate")
async def evaluate_model(
    evaluation_request: EvaluationRequest,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    """Start model evaluation as a background task.

    Computes cosine similarity metrics between question-positive-negative triplets
    to assess model quality. Optionally compares against a baseline model.

    Main metrics computed:
    - Average cosine similarity between question and positive examples
    - Average cosine similarity between question and negative examples
    - Ratio of positive to negative similarities
    - Spearman correlation for similarity ranking

    Returns:
        202 Accepted with Location header pointing to task status
    """
    if evaluation_request.dataset_id:
        try:
            UUID(evaluation_request.dataset_id)
        except ValueError as exc:
            raise InvalidDatasetIdError(evaluation_request.dataset_id) from exc

    task = EvaluationTask(id=uuid4())
    await save_evaluation_task_db(db, task)

    run_evaluation_bg.send(
        evaluation_request.model_dump(),
        str(task.id),
    )

    logger.debug(
        "Evaluation started in background",
        task_id=str(task.id),
        model_tag=evaluation_request.model_tag,
        dataset_id=evaluation_request.dataset_id,
    )

    location = f"/evaluation/{task.id}/status"
    return Response(
        status_code=status.HTTP_202_ACCEPTED,
        headers={"Location": location},
    )


@router.get("/{task_id}/status")
async def get_evaluation_status(
    task_id: UUID,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> EvaluationStatusResponse:
    """Get the status and results of an evaluation task."""
    task = await get_evaluation_task_by_id_db(db, task_id)
    if not task:
        raise EvaluationTaskNotFoundError(str(task_id))
    return EvaluationStatusResponse.from_task(task)


SELECTED_MTEB_TASKS = ["STSBenchmark", "BIOSSES", "SICK-R"]
CACHE_BASE_DIR = "/app/data/models"


@router.get("/mteb/{model_tag}", summary="Run MTEB benchmark on a cached model by UUID")
async def get_evaluation_results(
    model_tag: str,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> JSONResponse:
    """Run selected MTEB benchmark tasks on a locally cached model identified by tag.

    Args:
        model_tag: The tag or UUID identifying the model in the database.
        db: Injected asynchronous DB session.

    Returns:
        JSONResponse containing benchmark results or error details.
    """
    try:
        results = await EvaluationService(db).run_benchmark(model_tag)
        return JSONResponse(content=results)
    except Exception as e:
        logger.warning("Benchmark failed : {}", e)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Benchmark execution failed"
            },
        )
