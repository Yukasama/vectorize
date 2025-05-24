"""Main application module for the Vectorize service."""

from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Final

from aiofiles.os import makedirs
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Gauge
from prometheus_fastapi_instrumentator import Instrumentator
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.config import (
    add_security_headers,
    config_logger,
    engine,
    seed_db,
    settings,
)
from vectorize.utils.error_handler import register_exception_handlers
from vectorize.utils.routers import register_routers

config_logger()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    """Initialize resources on startup."""
    await makedirs(settings.dataset_upload_dir, exist_ok=True)
    await makedirs(settings.model_upload_dir, exist_ok=True)

    async with engine.begin() as conn:
        if settings.clear_db_on_restart:
            await conn.run_sync(SQLModel.metadata.drop_all)
        await conn.run_sync(SQLModel.metadata.create_all)

    if settings.app_env != "production":
        async with AsyncSession(engine) as session:
            await seed_db(session)

    yield
    await engine.dispose()


app: Final = FastAPI(
    title="Vectorize Service",
    description="Service for text embedding and vector operations",
    version="0.1.0",
    lifespan=lifespan,
)


# --------------------------------------------------------
# P R O M E T H E U S
# --------------------------------------------------------
Instrumentator().instrument(app).expose(app)


REQUESTS_IN_PROGRESS = Gauge(
    "http_requests_in_progress_total",
    "Active HTTP requests",
    ["method", "path"],
)


@app.middleware("http")
async def track_in_flight(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Request:
    """Middleware to track the number of in-flight requests."""
    REQUESTS_IN_PROGRESS.labels(request.method, request.url.path).inc()
    try:
        return await call_next(request)
    finally:
        REQUESTS_IN_PROGRESS.labels(request.method, request.url.path).dec()


# --------------------------------------------------------
# R O U T E R S
# --------------------------------------------------------
register_routers(app)


# --------------------------------------------------------
# C O R S
# --------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origin,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-Requested-With"],
    allow_credentials=False,
    max_age=600,
    expose_headers=["Location"],
)


# --------------------------------------------------------
# S E C U R I T Y
# --------------------------------------------------------
add_security_headers(app)


# --------------------------------------------------------
# E X C E P T I O N S
# --------------------------------------------------------
register_exception_handlers(app)
