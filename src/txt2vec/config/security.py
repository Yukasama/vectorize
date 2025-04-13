"""Security headers module."""

from fastapi import Response

__all__ = ["set_security_headers"]


def set_security_headers(response: Response) -> None:
    """Set strict security headers to harden the API."""
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = (
        "default-src 'none'; frame-ancestors 'none'"
    )

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-XSS-Protection"] = "1; mode=block"

    # response.headers["Strict-Transport-Security"] = (
    #     "max-age=63072000; includeSubDomains; preload"
    # )

    response.headers["Server"] = ""
    response.headers["X-Powered-By"] = ""

    response.headers["Permissions-Policy"] = (
        "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
        "magnetometer=(), microphone=(), payment=(), usb=(), interest-cohort=()"
    )
