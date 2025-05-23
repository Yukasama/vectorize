"""Upload module for vectorize."""

from vectorize.upload.router import router
from vectorize.upload.zip_service import upload_zip_model

__all__ = ["router", "upload_zip_model"]
