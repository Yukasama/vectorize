"""Get the path to the error handler module."""

import traceback

from vectorize.common.app_error import AppError, ETagError


def get_error_path(err: AppError | ETagError | Exception) -> str:
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
    filename, line, func, _ = error_location

    app_path = filename.split("vectorize")[-1] if "vectorize" in filename else filename
    filename = f"vectorize{app_path}"
    return f"{filename}:{line} (fn:{func})"
