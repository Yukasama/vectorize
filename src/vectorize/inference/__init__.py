"""Inference module for text embedding generation and model inference."""

from .embedding_model import EmbeddingData, EmbeddingUsage, Embeddings
from .repository import create_inference_counter, get_model_count
from .router import router
from .schemas import EmbeddingRequest
from .service import get_embeddings_srv, get_model_stats_srv

__all__ = [
    "EmbeddingData",
    "EmbeddingRequest",
    "EmbeddingUsage",
    "Embeddings",
    "create_inference_counter",
    "get_embeddings_srv",
    "get_model_count",
    "get_model_stats_srv",
    "router",
]
