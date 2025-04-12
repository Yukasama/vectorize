"""Security headers module."""

from fastapi import Response


def set_security_headers(response: Response) -> None:
    """Set strict security headers to harden the API."""
    # 1. Prevent clickjacking
    response.headers["X-Frame-Options"] = "DENY"

    # 2. Enforce a restrictive Content Security Policy
    #    - 'default-src \'none\'' => no resource is allowed to load by default
    #    - 'frame-ancestors \'none\'' => prevents embedding your site in any iframe
    response.headers["Content-Security-Policy"] = (
        "default-src 'none'; frame-ancestors 'none'"
    )

    # 3. Block sniffing of content-type
    response.headers["X-Content-Type-Options"] = "nosniff"

    # 4. Minimal legacy protection for older browsers against reflected XSS
    response.headers["X-XSS-Protection"] = "1; mode=block"

    # 5. Strict Transport Security (force HTTPS). Two years + preload + subdomains.
    # response.headers["Strict-Transport-Security"] = (
    #     "max-age=63072000; includeSubDomains; preload"
    # )

    # 6. Remove "Server" and "X-Powered-By" to avoid leaking server info
    response.headers["Server"] = ""
    response.headers["X-Powered-By"] = ""

    # 7. Lock down browser features: disallow use of sensors, camera, etc.
    response.headers["Permissions-Policy"] = (
        "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
        "magnetometer=(), microphone=(), payment=(), usb=(), interest-cohort=()"
    )
