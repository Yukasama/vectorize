"""Inference Router."""

from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    Response,
)
from sqlmodel.ext.asyncio.session import AsyncSession

from txt2vec.config.db import get_session

from .embedding_model import Embeddings
from .request_model import EmbeddingRequest
from .service import create_embeddings

__all__ = ["router"]


router = APIRouter(tags=["Embeddings", "Inference"])


@router.post("")
async def get_embeddings(
    data: EmbeddingRequest,
    response: Response,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> Embeddings:
    """Get embeddings for an input text.

    Args:
        data: The embedding request containing input text and model specifications.
        response: FastAPI response object for setting headers.
        db: Database session for retrieving model information.

    Returns:
        Embeddings: OpenAI-compatible response containing the generated
            embeddings and usage statistics.

    Raises:
        ModelNotFoundError: If the requested model cannot be found.
        ModelLoadError: If there's an error loading the AI model.
    """
    embeddings = await create_embeddings(db, data)
    response.headers["ETAG"] = "0"
    return embeddings
