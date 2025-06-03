"""Save dataset as CSV file to disk."""

from pathlib import Path

import pandas as pd

from vectorize.config import settings

__all__ = ["_delete_dataset_from_fs", "_save_dataframe_to_fs"]


def _save_dataframe_to_fs(df: pd.DataFrame, filename: str) -> Path:
    """Persist DataFrame as CSV in upload_dir and return its path.

    Args:
        df: DataFrame to write.
        filename: Target filename (already sanitised).

    Returns:
        Path pointing to the saved CSV file.
    """
    out_path = settings.dataset_upload_dir / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def _delete_dataset_from_fs(filename: str) -> bool:
    """Delete dataset file from filesystem.

    Args:
        filename: Target filename to delete (already sanitised).

    Returns:
        True if file was deleted, False if file didn't exist.

    Raises:
        PermissionError: If file cannot be deleted due to permissions.
        OSError: If deletion fails for other reasons.
    """
    file_path = settings.dataset_upload_dir / filename

    if not file_path.exists():
        return False

    if not file_path.is_file():
        raise OSError(f"Path exists but is not a file: {file_path}")

    file_path.unlink()
    return True
