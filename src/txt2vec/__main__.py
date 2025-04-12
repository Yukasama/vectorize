"""For python -m txt2vec, so that the app can be run without uv."""

from txt2vec.server import run

__all__ = ["run"]

if __name__ == "__main__":
    run()
