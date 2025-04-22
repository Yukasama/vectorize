"""Main application module for the Text2Vec service."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Final

from aiofiles.os import makedirs
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from txt2vec.config import (
    add_security_headers,
    close_db,
    config_logger,
    init_db,
)
from txt2vec.config.config import (
    dataset_upload_dir,
    model_upload_dir,
    prefix,
)
from txt2vec.config.seed import seed_db
from txt2vec.datasets.router import router as dataset_router
from txt2vec.handle_exceptions import handle_exception
from txt2vec.upload.router import router as upload_router

config_logger()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    """Initialize resources on startup."""
    await init_db()
    await seed_db()
    await makedirs(dataset_upload_dir, exist_ok=True)
    await makedirs(model_upload_dir, exist_ok=True)
    yield
    logger.info("Server being shutdown...")
    await close_db()


app: Final = FastAPI(
    title="Text2Vec Service",
    description="Service for text embedding and vector operations",
    version="2025.4.1",
    lifespan=lifespan,
)


# --------------------------------------------------------
# R O U T E R S
# --------------------------------------------------------
base_router = APIRouter(prefix=prefix)
base_router.include_router(dataset_router, prefix="/datasets")
base_router.include_router(upload_router, prefix="/uploads")

app.include_router(base_router)


# --------------------------------------------------------
# S E C U R I T Y
# --------------------------------------------------------
add_security_headers(app)


@app.exception_handler(Exception)
def global_handler(request: Request, exc: Exception):
    logger.opt(exception=True).error("Global error caught: {}", str(exc))
    http_exc = handle_exception(exc)
    return JSONResponse(status_code=http_exc.status_code, content=http_exc.detail)
