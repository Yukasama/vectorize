"""Get the path to the error handler module."""

import traceback

from vectorize.common.app_error import AppError


def get_error_path(err: AppError) -> str:
    """Extract formatted source location from AppError traceback.

    Extracts and formats the source code location where an AppError occurred,
    focusing on the vectorize package path. Formats the output as a path string
    with file location, line number, and function name.

    Args:
        err: The application error containing traceback information

    Returns:
        A formatted string containing the error's source location in the format:
        "filename:line (fn:function_name)"
    """
    error_location = traceback.extract_tb(err.__traceback__)[-1]
    full_filename, line, func, _ = error_location

    vectorize_path = (
        full_filename.split("vectorize")[-1]
        if "vectorize" in full_filename
        else full_filename
    )
    filename = f"vectorize{vectorize_path}"
    return f"{filename}:{line} (fn:{func})"
