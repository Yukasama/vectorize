"""Upload module for txt2vec."""

from txt2vec.upload.router import router
from txt2vec.upload.zip_service import upload_zip_model

__all__ = ["router", "upload_zip_model"]
