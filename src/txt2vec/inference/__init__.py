"""Inference module for text embedding generation and model inference."""

from .embedding_model import EmbeddingData, EmbeddingUsage, Embeddings
from .repository import create_inference_counter, get_model_count
from .request_model import EmbeddingRequest
from .router import router
from .service import get_embeddings_srv, get_model_stats_srv
from .utils.generator import _generate_embeddings
from .utils.model_loader import _load_model
from .utils.pool_mean import _mean_pool

__all__ = [
    "EmbeddingData",
    "EmbeddingRequest",
    "EmbeddingUsage",
    "Embeddings",
    "_generate_embeddings",
    "_load_model",
    "_mean_pool",
    "create_inference_counter",
    "get_embeddings_srv",
    "get_model_count",
    "get_model_stats_srv",
    "router",
]
