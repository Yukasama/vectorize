"""Main application module for the Text2Vec service."""

from collections.abc import Awaitable, Callable
from typing import Any, Final

from fastapi import FastAPI, Request, Response

from txt2vec.config.security import set_security_headers

app: Final = FastAPI(
    title="Text2Vec Service",
    description="Service for text embedding and vector operations",
    version="2025.4.1",
)


@app.get("/")
def read_root() -> dict[str, str]:
    """Root endpoint for the FastAPI application.

    :return: A simple greeting message
    :rtype: dict[str, str]
    """
    return {"message": "Welcome to the Text2Vec API!"}


@app.middleware("http")
async def add_security_headers(
    request: Request,
    call_next: Callable[[Any], Awaitable[Response]],
) -> Response:
    """Add security headers to all HTTP responses.

    Intercepts responses and applies security headers to protect against
    common web vulnerabilities.

    :param request: The incoming HTTP request
    :param call_next: The next middleware in the chain
    :return: The HTTP response with added security headers
    """
    response: Final[Response] = await call_next(request)
    set_security_headers(response)
    return response
