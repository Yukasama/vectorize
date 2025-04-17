"""Main application module for the Text2Vec service."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Final

from aiofiles.os import makedirs
from fastapi import APIRouter, FastAPI
from loguru import logger

from txt2vec.config import close_db, init_db
from txt2vec.config.config import app_config
from txt2vec.config.logger import config_logger
from txt2vec.config.security import add_security_headers
from txt2vec.datasets.router import router as dataset_router
from txt2vec.upload.router import router as upload_router

server_config = app_config.get("server", {})
dataset_config = app_config.get("dataset", {})

config_logger()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    """Initialize resources on startup."""
    await init_db()
    await makedirs(dataset_config.get("upload_dir"), exist_ok=True)
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
v1_router = APIRouter(prefix=f"/{server_config.get('prefix', 'v1')}")

v1_router.include_router(dataset_router, prefix="/datasets")
v1_router.include_router(upload_router, prefix="/uploads")

app.include_router(v1_router)


# --------------------------------------------------------
# S E C U R I T Y
# --------------------------------------------------------
add_security_headers(app)
