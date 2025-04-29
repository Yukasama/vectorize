"""Inference service."""

from sqlmodel.ext.asyncio.session import AsyncSession

from txt2vec.ai_model.repository import get_ai_model

from .embedding_model import EmbeddingUsage, Embeddings
from .request_model import EmbeddingRequest
from .utils.generator import generate_embeddings
from .utils.model_loader import load_model

__all__ = ["create_embeddings"]


async def create_embeddings(db: AsyncSession, data: EmbeddingRequest) -> Embeddings:
    """Generate embeddings for input text using the specified model.

    Loads the AI model specified by the client, processes the input data,
    and creates vector embeddings for the provided text or token arrays.

    Args:
        db: Database session for retrieving model information.
        data: The embedding request containing input text and model specifications.

    Returns:
        Embeddings: OpenAI-compatible response containing the generated
            embeddings and usage statistics.

    Raises:
        ModelNotFoundError: If the requested model cannot be found.
        ModelLoadError: If there's an error loading the AI model.
    """
    ai_model = await get_ai_model(db, data.model)
    model, tokenizer = load_model(ai_model.model_tag)
    results, total_toks = generate_embeddings(data, model, tokenizer)

    return Embeddings(
        object="list",
        data=results,
        model=data.model,
        usage=EmbeddingUsage(prompt_tokens=total_toks, total_tokens=total_toks),
    )
