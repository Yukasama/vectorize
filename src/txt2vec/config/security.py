"""Security headers."""

import os
from collections.abc import Awaitable, Callable
from typing import Any, Final

from fastapi import FastAPI, Request, Response

__all__ = ["add_security_headers"]

ENVIRONMENT = os.getenv("APP_ENV", "development").lower()


def _set_security_headers(response: Response) -> None:
    """Set security headers to harden the API.

    :param response: The HTTP response to modify
    """
    response.headers["X-Frame-Options"] = "DENY"

    if ENVIRONMENT == "production":
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "img-src 'self' data:; "
            "style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; "
            "frame-ancestors 'none'"
        )
    else:
        response.headers["Content-Security-Policy"] = (
            "default-src * 'unsafe-inline' 'unsafe-eval'; img-src * data:"
        )

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-XSS-Protection"] = "1; mode=block"

    # Uncomment for HTTPS - leave commented for local development
    # response.headers["Strict-Transport-Security"] = (
    #     "max-age=63072000; includeSubDomains; preload"
    # )

    response.headers["Server"] = ""
    response.headers["X-Powered-By"] = ""

    response.headers["Permissions-Policy"] = (
        "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
        "magnetometer=(), microphone=(), payment=(), usb=(), interest-cohort=()"
    )


def add_security_headers(app: FastAPI) -> None:
    """Configure application middleware."""

    @app.middleware("http")
    # pylint: disable=unused-function
    # pyright: ignore
    async def _security_headers_middleware(
        request: Request,
        call_next: Callable[[Any], Awaitable[Response]],
    ) -> Response:
        """Add security headers to all HTTP responses."""
        response: Final[Response] = await call_next(request)
        _set_security_headers(response)
        return response
