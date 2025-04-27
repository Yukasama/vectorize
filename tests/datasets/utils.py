"""Utils for datasets tests."""

from pathlib import Path

__all__ = ["get_test_file"]


def get_test_file(file_path: Path) -> dict[str, tuple[str, bytes, str]]:
    """Read file content and determine MIME type based on extension."""
    content = file_path.read_bytes()
    ext = file_path.suffix.lower()

    mime_types = {
        ".csv": "text/csv",
        ".json": "application/json",
        ".xml": "application/xml",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
    mime_type = mime_types.get(ext, "application/octet-stream")

    return {"file": (file_path.name, content, mime_type)}
