"""Main application module for the Text2Vec service."""

from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, Final

from aiofiles.os import makedirs
from fastapi import FastAPI, Request, Response
from loguru import logger

from txt2vec.config import UPLOAD_DIR, set_security_headers
from txt2vec.datasets.router import router


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    """Initialize resources on startup."""
    await makedirs(UPLOAD_DIR, exist_ok=True)
    yield
    logger.info("Server being shutdown...")


app: Final = FastAPI(
    title="Text2Vec Service",
    description="Service for text embedding and vector operations",
    version="2025.4.1",
    root_path="/v1",
    lifespan=lifespan,
)


@app.get("/")
def read_root() -> dict[str, str]:
    """Root endpoint for the FastAPI application.

    :return: A simple greeting message
    :rtype: dict[str, str]
    """
    return {"message": "Welcome to the Text2Vec API!"}


# --------------------------------------------------------
# R O U T E R S
# --------------------------------------------------------
app.include_router(router)


# --------------------------------------------------------
# S E C U R I T Y
# --------------------------------------------------------
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
