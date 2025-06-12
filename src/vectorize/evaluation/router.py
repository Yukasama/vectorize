"""Router for model evaluation endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.config.db import get_session

from .schemas import EvaluationRequest, EvaluationResponse
from .service import evaluate_model_task

__all__ = ["router"]

router = APIRouter(tags=["Evaluation"])


@router.post("/evaluate")
async def evaluate_model(
    evaluation_request: EvaluationRequest,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> EvaluationResponse:
    """Evaluate a trained model on a dataset.
    
    Computes cosine similarity metrics between question-positive-negative triplets
    to assess training quality. Optionally compares against a baseline model.
    
    Main metrics computed:
    - Average cosine similarity between question and positive examples
    - Average cosine similarity between question and negative examples  
    - Ratio of positive to negative similarities (should be > 1)
    - Spearman correlation for similarity ranking
    
    A training is considered successful if:
    - Positive similarities > negative similarities
    - Similarity ratio > 1.2
    """
    return await evaluate_model_task(db, evaluation_request)
