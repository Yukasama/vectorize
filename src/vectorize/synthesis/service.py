"""Synthesis service layer."""

from uuid import UUID

from fastapi import HTTPException, status

__all__ = ["validate_upload_request"]


def validate_upload_request(
    files: list | None, existing_dataset_id: UUID | None
) -> None:
    """Validate that either files or dataset ID is provided.

    Args:
        files: List of uploaded files
        existing_dataset_id: Optional existing dataset ID

    Raises:
        HTTPException: If neither files nor dataset ID provided
    """
    if not files and existing_dataset_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either files or existing dataset ID must be provided.",
        )
