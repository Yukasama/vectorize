"""Common module for shared utilities and error handling.

This module provides foundational components used throughout the application,
including error handling, exceptions, and shared utilities for consistent behavior
across all modules.

Key Components:
- Error handling: Centralized error classes and HTTP exception mapping
- App errors: Application-specific error types with structured error codes
- HTTP exceptions: RESTful API error responses with proper status codes
- Version management: ETag-based versioning for resource management

The common module ensures consistent error handling patterns, standardized
HTTP responses, and shared utilities that maintain application-wide conventions
for logging, validation, and error reporting.
"""

from .app_error import AppError, ETagError
from .exceptions import (
    InternalServerError,
    InvalidFileError,
    NotFoundError,
    VersionMismatchError,
    VersionMissingError,
)

__all__ = [
    "AppError",
    "ETagError",
    "InternalServerError",
    "InvalidFileError",
    "NotFoundError",
    "VersionMismatchError",
    "VersionMissingError",
]
