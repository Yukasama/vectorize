"""Declaration of the root package txt2vec."""

from txt2vec.app import app
from txt2vec.server import run

__all__ = ["app", "main"]


def main() -> None:
    """Run the application server using uv."""
    run()
