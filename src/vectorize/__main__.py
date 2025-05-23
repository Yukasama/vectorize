"""For python -m vectorize, so that the app can be run without having uv installed."""

from vectorize.server import run

__all__ = ["run"]

if __name__ == "__main__":
    run()
