"""Get the path to the error handler module."""

import traceback

from txt2vec.common.app_error import AppError


def get_error_path(err: AppError) -> str:
    """Extract formatted source location from AppError traceback.

    Extracts and formats the source code location where an AppError occurred,
    focusing on the txt2vec package path. Formats the output as a path string
    with file location, line number, and function name.

    Args:
        err: The application error containing traceback information

    Returns:
        A formatted string containing the error's source location in the format:
        "filename:line (fn:function_name)"
    """
    error_location = traceback.extract_tb(err.__traceback__)[-1]
    full_filename, line, func, _ = error_location

    txt2vec_path = (
        full_filename.split("txt2vec")[-1]
        if "txt2vec" in full_filename
        else full_filename
    )
    filename = f"txt2vec{txt2vec_path}"
    return f"{filename}:{line} (fn:{func})"
