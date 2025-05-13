"""Global exception handlers for Application."""

from asyncio import CancelledError

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from loguru import logger

from txt2vec.common.app_error import AppError, ETagError
from txt2vec.config import settings
from txt2vec.config.errors import ErrorCode, ErrorNames

from .error_path import get_error_path

__all__ = ["register_exception_handlers"]


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all global exception handlers to the FastAPI application.

    Registers handlers for:
    - Application errors (AppError)
    - ETag errors (ETagError)
    - Validation errors (RequestValidationError)
    - Unexpected exceptions (ServerError)

    Args:
        app: The FastAPI application instance to register handlers with.
    """

    @app.exception_handler(AppError)
    def _handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
        """Handle application-specific errors.

        Args:
            request: The incoming HTTP request.
            exc: The application error that was raised.

        Returns:
            JSONResponse: A formatted error response with appropriate status code.
        """
        logger.debug("{}: {}", exc.error_code, exc.message, path=get_error_path(exc))
        return _make_response(exc.status_code, exc.error_code, exc.message)

    @app.exception_handler(ETagError)
    def _handle_etag_error(_request: Request, exc: ETagError) -> JSONResponse:
        """Handle application-specific errors.

        Args:
            request: The incoming HTTP request.
            exc: The application error that was raised.

        Returns:
            JSONResponse: A formatted error response with appropriate status code.
        """
        logger.debug("{}: {}", exc.error_code, exc.message, path=get_error_path(exc))
        return _make_response(exc.status_code, exc.error_code, exc.message, exc.version)

    @app.exception_handler(Exception)
    def _handle_unexpected(_request: Request, exc: Exception) -> JSONResponse:
        """Handle any uncaught exceptions as 500 server errors.

        Args:
            request: The incoming HTTP request.
            exc: The uncaught exception.

        Returns:
            JSONResponse: A 500 error response.

        Raises:
            CancelledError: Re-raised in non-development environments.
        """
        # Pass through cancellations in dev
        if isinstance(exc, CancelledError) and settings.app_env != "development":
            raise exc

        logger.exception("{}", str(exc), path=get_error_path(exc))
        return _make_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            ErrorCode.SERVER_ERROR,
            ErrorNames.INTERNAL_SERVER_ERROR,
        )


def _make_response(
    status_code: int,
    code: ErrorCode,
    message: str,
    version: int | None = None,
) -> JSONResponse:
    """Create a standardized JSON error response.

    Args:
        status_code: HTTP status code to return.
        code: Application-specific error code enum value.
        message: Human-readable error message.
        version: Optional ETag header value for versioning.

    Returns:
        JSONResponse: A formatted JSON response with the error details.
    """
    headers = {}
    if version is not None:
        headers["ETag"] = f'"{version}"'

    return JSONResponse(
        status_code=status_code,
        content={"code": code, "message": message},
        headers=headers,
    )
