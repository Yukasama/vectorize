"""Parse ETag header."""

from fastapi import Request

from vectorize.common.exceptions import VersionMissingError

__all__ = ["parse_etag"]


def parse_etag(resource_id: str, request: Request) -> int:
    """Return integer version from *If-Match* header or raise."""
    value = request.headers.get("If-Match", "").strip()
    if not (value.startswith('"') and value.endswith('"')):
        raise VersionMissingError(resource_id)

    try:
        return int(value.strip('"'))
    except ValueError as exc:
        raise VersionMissingError(resource_id) from exc
