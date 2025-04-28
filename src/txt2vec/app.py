"""Main application module for the Text2Vec service."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Final

from aiofiles.os import makedirs
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from txt2vec.config import (
    add_security_headers,
    config_logger,
)
from txt2vec.config.config import (
    allow_origin,
    dataset_upload_dir,
    model_upload_dir,
    prefix,
)
from txt2vec.config.db import engine
from txt2vec.config.seed import seed_db
from txt2vec.datasets.router import router as dataset_router
from txt2vec.error_handler import register_exception_handlers
from txt2vec.inference.router import router as embeddings_router
from txt2vec.upload.router import router as upload_router

config_logger()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    """Initialize resources on startup."""
    await makedirs(dataset_upload_dir, exist_ok=True)
    await makedirs(model_upload_dir, exist_ok=True)

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
        await conn.run_sync(SQLModel.metadata.create_all)

    async with AsyncSession(engine) as session:
        await seed_db(session)

    yield

    logger.info("Server being shutdown...")
    await engine.dispose()


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
base_router.include_router(embeddings_router, prefix="/embeddings")

app.include_router(base_router)


# --------------------------------------------------------
# C O R S
# --------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origin,
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
