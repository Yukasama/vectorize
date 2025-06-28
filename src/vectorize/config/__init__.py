"""Configuration module for the Vectorize application.

This module provides centralized configuration management for the entire application,
including database connections, logging setup, error handling, and application settings.

Key Components:
- settings: Application configuration loaded from environment variables and TOML files
- Database: SQLAlchemy engine and session management for SQLite
- Logging: Loguru-based logging configuration with development/production modes
- Error handling: Centralized error codes and exception definitions
- Database seeding: Initial data setup for development and testing

The configuration supports multiple environments (development, testing, production)
and allows runtime configuration through environment variables while maintaining
sensible defaults from the app.toml configuration file.
"""

from vectorize.config.config import settings
from vectorize.config.db import engine, get_session
from vectorize.config.errors import ErrorCode, ErrorNames
from vectorize.config.logger import config_logger
from vectorize.config.seed import seed_db

__all__ = [
    "ErrorCode",
    "ErrorNames",
    "config_logger",
    "engine",
    "get_session",
    "seed_db",
    "settings",
]
