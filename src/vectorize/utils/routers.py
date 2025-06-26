"""Router Initializer."""

from fastapi import FastAPI

from vectorize.ai_model.router import router as models_router
from vectorize.common.router import router as common_router
from vectorize.dataset.router import router as dataset_router
from vectorize.evaluation.router import router as evaluation_router
from vectorize.inference.router import router as embeddings_router
from vectorize.synthesis.router import router as synthesis_router
from vectorize.task.router import router as task_router
from vectorize.training.router import router as training_router
from vectorize.upload.router import router as upload_router


def register_routers(app: FastAPI) -> None:
    """Register all API routers with the FastAPI application.

    Args:
        app: FastAPI application instance.
    """
    app.include_router(common_router)
    app.include_router(dataset_router, prefix="/datasets")
    app.include_router(upload_router, prefix="/uploads")
    app.include_router(embeddings_router, prefix="/embeddings")
    app.include_router(models_router, prefix="/models")
    app.include_router(evaluation_router, prefix="/evaluation")
    app.include_router(synthesis_router, prefix="/synthesis")
    app.include_router(task_router, prefix="/tasks")
    app.include_router(training_router, prefix="/training")
