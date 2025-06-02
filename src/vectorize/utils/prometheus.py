"""Prometheus metrics for tracking custom metrics."""

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from prometheus_client import Gauge

REQUESTS_IN_PROGRESS = Gauge(
    "http_requests_in_progress_total",
    "Active HTTP requests",
    ["method", "path"],
)


def add_prometheus_metrics(app: FastAPI) -> None:
    """Configure the FastAPI application to track in-flight HTTP requests.

    Args:
        app: The FastAPI application instance where the middleware will be added.
    """

    @app.middleware("http")
    async def track_in_flight(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Middleware to track the number of in-flight requests."""
        REQUESTS_IN_PROGRESS.labels(request.method, request.url.path).inc()
        try:
            return await call_next(request)
        finally:
            REQUESTS_IN_PROGRESS.labels(request.method, request.url.path).dec()
