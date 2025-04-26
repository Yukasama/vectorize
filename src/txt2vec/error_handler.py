"""Global exception handlers for Application."""

import asyncio

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger

from txt2vec.config.config import app_env
from txt2vec.errors import AppError, ErrorCode


def _make_response(
    status_code: int,
    code: ErrorCode,
    message: str,
) -> JSONResponse:
    """Return a uniform JSON error shape."""
    return JSONResponse(
        status_code=status_code,
        content={"code": code, "message": message},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all global exception handlers to *app*."""

    # --- Domain errors ------------------------------------------------
    @app.exception_handler(AppError)
    def _handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        logger.error("{}: {}", exc.error_code, exc.message)
        return _make_response(exc.status_code, exc.error_code, exc.message)

    # --- Validation errors -------------------------------------------------
    @app.exception_handler(RequestValidationError)
    def _handle_validation(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.info("Validation error on {}: {}", request.url, exc.errors())
        return _make_response(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            ErrorCode.VALIDATION_ERROR,
            "Validation failed",
        )

    # --- Catch-all barrier ------------------------------
    @app.exception_handler(Exception)
    def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        # Pass through cancellations in dev
        if isinstance(exc, asyncio.CancelledError) and app_env != "development":
            raise exc

        logger.opt(exception=True).error("Unhandled exception: {}", str(exc))
        return _make_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            ErrorCode.SERVER_ERROR,
            "Internal server error",
        )
