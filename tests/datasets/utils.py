"""Utils for datasets tests."""

from pathlib import Path

__all__ = ["build_files"]


def build_files(path: Path) -> list[tuple[str, tuple[str, bytes, str]]]:
    """Create file upload format for API tests.

    Prepares a file for upload by creating the tuple structure expected
    by the FastAPI TestClient for file uploads.

    Args:
        path: Path object to the file to upload

    Returns:
        List containing a tuple with form field name and file details
        in the format expected by TestClient
    """
    return [("files", (path.name, path.read_bytes(), "application/octet-stream"))]
