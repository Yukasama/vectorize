"""Get the path to the error handler module."""

import traceback

from txt2vec.errors import AppError


def get_error_path(err: AppError) -> str:
    """Get the path to the error handler module."""
    error_location = traceback.extract_tb(err.__traceback__)[-1]
    full_filename, line, func, _ = error_location

    txt2vec_path = (
        full_filename.split("txt2vec")[-1]
        if "txt2vec" in full_filename
        else full_filename
    )
    filename = f"txt2vec{txt2vec_path}"
    return f"{filename}:{line} (fn:{func})"
