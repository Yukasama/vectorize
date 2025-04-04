import sys
import uvicorn


def main():
    """Entry point for the application."""
    uvicorn.run(
        "txt2vec_service.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["src"],
    )


if __name__ == "__main__":
    sys.exit(main())