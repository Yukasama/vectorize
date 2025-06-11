"""Inference service."""

from collections import defaultdict
from datetime import UTC, datetime, timedelta

from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.ai_model.repository import get_ai_model_db
from vectorize.model_loader import load_model  # ← Neuer Import aus Top-Level

from .embedding_model import EmbeddingUsage, Embeddings
from .repository import create_inference_counter, get_model_count
from .schemas import EmbeddingRequest
from .utils.generator import _generate_embeddings  # ← Lokaler Import

__all__ = ["get_embeddings_srv", "get_model_stats_srv"]


async def get_embeddings_srv(db: AsyncSession, data: EmbeddingRequest) -> Embeddings:
    """Generate embeddings for input text using the specified model."""
    ai_model = await get_ai_model_db(db, data.model)

    # Verwendet jetzt den zentralen model_loader
    model, tokenizer = load_model(ai_model.model_tag)
    results, total_toks = _generate_embeddings(data, model, tokenizer)

    await create_inference_counter(db, ai_model.id)

    return Embeddings(
        object="list",
        data=results,
        model=data.model,
        usage=EmbeddingUsage(prompt_tokens=total_toks, total_tokens=total_toks),
    )


async def get_model_stats_srv(db: AsyncSession, model_tag: str) -> dict[str, int]:
    """Get daily inference statistics for an AI model."""
    ai_model = await get_ai_model_db(db, model_tag)
    creation_date = ai_model.created_at.date()
    counters = await get_model_count(db, model_tag)

    daily_counts = defaultdict(int)
    for counter in counters:
        counter_date = counter.created_at.date().isoformat()
        daily_counts[counter_date] += 1

    end_date = datetime.now(tz=UTC).date()

    result = {}
    current_date = creation_date
    while current_date <= end_date:
        date_str = current_date.isoformat()
        result[date_str] = daily_counts.get(date_str, 0)
        current_date += timedelta(days=1)

    return result
