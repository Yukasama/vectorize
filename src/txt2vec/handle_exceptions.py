from typing import Callable
from functools import wraps
from fastapi import HTTPException
from loguru import logger
from txt2vec.services.exceptions import handle_dataset_exception

def handle_exceptions(func: Callable) -> Callable:
    """
    Decorator for standardized exception handling in API endpoints.

    This decorator catches all exceptions from the endpoint function,
    logs them, and converts them to proper HTTPExceptions with
    standardized error formats.
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in endpoint {func.__name__}: {str(e)}")
            http_exception = handle_dataset_exception(e)
            raise http_exception

    return wrapper
