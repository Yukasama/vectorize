"""Define configuration for the project."""

from pathlib import Path
from typing import Final

__all__ = ["BASE_URL", "UPLOAD_DIR"]


UPLOAD_DIR: Final = Path("data/uploads")
BASE_URL = "http://localhost:8000/v1/"
