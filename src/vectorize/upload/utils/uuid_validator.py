"""utils for validating UUIDs in task IDs."""
from uuid import UUID

from vectorize.upload.exceptions import InvalidUUIDError

__all__ = ["parse_uuid"]


def parse_uuid(task_id: str) -> UUID:
    """Parse a string into a UUID.

    Args:
        task_id: The string representation of the UUID.

    Raises:
        InvalidUUIDError if the string is not a valid UUID.
    """
    try:
        return UUID(task_id)
    except ValueError as err:
        raise InvalidUUIDError() from err
