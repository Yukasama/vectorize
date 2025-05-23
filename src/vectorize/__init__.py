"""Declaration of the root package vectorize."""

from vectorize.app import app
from vectorize.server import run

__all__ = ["app", "main"]


def main() -> None:
    """Run the application server using uv."""
    run()
