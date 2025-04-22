"""Upload module for txt2vec."""

from txt2vec.upload.router import router
from upload.local_service import upload_embedding_model

__all__ = ["router", "upload_embedding_model"]