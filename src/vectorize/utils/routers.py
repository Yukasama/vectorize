"""Router Initializer."""

from fastapi import APIRouter, FastAPI
from loguru import logger

from vectorize.ai_model.router import router as models_router
from vectorize.common.router import router as common_router
from vectorize.config.config import settings
from vectorize.dataset.router import router as dataset_router
from vectorize.evaluation.router import router as evaluation_router
from vectorize.inference.router import router as embeddings_router
from vectorize.synthesis.router import router as synthesis_router
from vectorize.task.router import router as task_router
from vectorize.upload.router import router as upload_router

# Import training router with better error handling
training_router = None
try:
    from vectorize.training.router import router as training_router
    logger.info("Training router imported successfully")
except ImportError as e:
    logger.error(f"Failed to import training router: {e}")
    logger.error("Training endpoints will not be available")
except Exception as e:
    logger.error(f"Unexpected error importing training router: {e}")
    logger.error("Training endpoints will not be available")


def register_routers(app: FastAPI) -> None:
    """Register all API routers with the FastAPI application.

    Args:
        app: FastAPI application instance.
    """
    app.include_router(common_router)

    base_router = APIRouter(prefix=settings.prefix)
    base_router.include_router(dataset_router, prefix="/datasets")
    base_router.include_router(upload_router, prefix="/uploads")
    base_router.include_router(embeddings_router, prefix="/embeddings")
    base_router.include_router(models_router, prefix="/models")
    
    # Only include training router if it was successfully imported
    if training_router is not None:
        base_router.include_router(training_router, prefix="/training")
        logger.info("Training router registered successfully")
    else:
        logger.warning("Training router not available - skipping registration")
    
    base_router.include_router(evaluation_router, prefix="/evaluation")
    base_router.include_router(synthesis_router, prefix="/synthesis")
    base_router.include_router(task_router, prefix="/tasks")

    app.include_router(base_router)
