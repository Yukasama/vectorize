"""Security headers."""

from collections.abc import Awaitable, Callable
from typing import Any, Final

from fastapi import FastAPI, Request, Response

from .config import settings

__all__ = ["add_security_headers"]


def add_security_headers(app: FastAPI) -> None:
    """Configure the FastAPI application to include security headers in all responses.

    Args:
        app: The FastAPI application instance where the middleware will be added.
    """

    @app.middleware("http")
    async def _security_headers_middleware(
        request: Request,
        call_next: Callable[[Any], Awaitable[Response]],
    ) -> Response:
        """Middleware to add security headers to every HTTP response.

        Args:
            request: The incoming HTTP request.
            call_next: A callable to send the request to the subsequent middleware or
            endpoint.

        Returns:
            Response: The HTTP response with security headers applied.
        """
        response: Final[Response] = await call_next(request)
        _set_security_headers(response)
        return response


def _set_security_headers(response: Response, path: str = "") -> None:
    """Set security headers to harden the API.

    Args:
        response: The HTTP response to secure
        path: The request path to determine if it's a documentation page
    """
    # Prevents the page from being displayed in frames to protect against clickjacking
    response.headers["X-Frame-Options"] = "DENY"

    is_docs = path.startswith(("/docs", "/redoc"))
    if settings.app_env == "production" and not is_docs:
        # Restricts resources the page can load to prevent XSS in production except docs
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "img-src 'self' data:; "
            "style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; "
            "frame-ancestors 'none'"
        )
    else:
        # Less restrictive CSP for development environments
        response.headers["Content-Security-Policy"] = (
            "default-src * 'unsafe-inline' 'unsafe-eval'; img-src * data:"
        )

    # Prevents browsers from interpreting files as a different MIME type
    response.headers["X-Content-Type-Options"] = "nosniff"

    # Enables browser's XSS filter to block rather than sanitize detected attacks
    response.headers["X-XSS-Protection"] = "1; mode=block"

    # Forces HTTPS connections for extended periods (commented for local development)
    # response.headers["Strict-Transport-Security"] = (
    #     "max-age=63072000; includeSubDomains; preload"
    # )

    # Removes server technology information to reduce attack surface
    response.headers["Server"] = ""
    response.headers["X-Powered-By"] = ""

    # Restricts which browser features and APIs can be used by the application
    response.headers["Permissions-Policy"] = (
        "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
        "magnetometer=(), microphone=(), payment=(), usb=(), interest-cohort=()"
    )
