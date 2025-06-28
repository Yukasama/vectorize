"""Common router."""

from fastapi import APIRouter, Response

__all__ = ["router"]


router = APIRouter(tags=["Common", "Health"])


@router.get("/", include_in_schema=False, summary="Root endpoint")
async def root() -> Response:
    """Root endpoint."""
    return Response("OK")


@router.get("/health", include_in_schema=False, summary="Health check endpoint")
async def health() -> Response:
    """Health check endpoint."""
    return Response(status_code=204)
