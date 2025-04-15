"""Upload router."""

from fastapi import APIRouter

router = APIRouter(tags=["Model Upload"])


@router.get("/")
def helloworld() -> dict[str, str]:
    """Root endpoint for the FastAPI application.

    :return: A simple greeting message
    :rtype: dict[str, str]
    """
    return {"message": "Hello World!"}
