"""Common router."""

from fastapi import APIRouter, Response

router = APIRouter(tags=["Common", "Health"])


@router.get("/", include_in_schema=False)
async def root() -> Response:
    """Root endpoint."""
    return Response("OK")


@router.get("/health", include_in_schema=False)
async def health() -> Response:
    """Health check endpoint."""
    return Response(status_code=204)
