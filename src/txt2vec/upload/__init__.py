"""Upload module for txt2vec."""

from txt2vec.upload.local_service import upload_embedding_model
from txt2vec.upload.router import router

__all__ = ["router", "upload_embedding_model"]
