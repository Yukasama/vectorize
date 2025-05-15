"""Inference module for text embedding generation and model inference."""

from .embedding_model import EmbeddingData, EmbeddingUsage, Embeddings
from .repository import create_inference_counter, get_model_count
from .request_model import EmbeddingRequest
from .router import router
from .service import get_embeddings_srv, get_model_stats_srv
from .utils.generator import generate_embeddings
from .utils.model_loader import load_model
from .utils.pool_mean import mean_pool

__all__ = [
    "EmbeddingData",
    "EmbeddingRequest",
    "EmbeddingUsage",
    "Embeddings",
    "create_inference_counter",
    "generate_embeddings",
    "get_embeddings_srv",
    "get_model_count",
    "get_model_stats_srv",
    "load_model",
    "mean_pool",
    "router",
]
