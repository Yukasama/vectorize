"""Define configuration for the project."""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["settings"]


_app_config_path = Path(__file__).parent / "resources" / "app.toml"
with Path.open(_app_config_path, "rb") as f:
    _config = tomllib.load(f)
    _app_config = _config.get("app", {})
    _server_config = _app_config.get("server", {})
    _dataset_config = _app_config.get("dataset", {})
    _model_config = _app_config.get("model", {})
    _inference_config = _app_config.get("inference", {})
    _db_config = _app_config.get("db", {})
    _log_config = _app_config.get("logging", {})


class Settings(BaseSettings):
    """Application configuration settings."""

    # Environment configuration
    app_env: Literal["development", "testing", "production"] = Field(
        default="development",
        description="Current application environment determining behavior.",
        validation_alias="ENV",
    )

    # Server configuration
    host_binding: str = Field(
        default=_server_config.get("host_binding", "127.0.0.1"),
        description="Host address the server binds to.",
    )

    port: int = Field(
        default=_server_config.get("port", 8000),
        description="Network port the server listens on.",
    )

    prefix: str = Field(
        default=_server_config.get("prefix", ""),
        description="API URL prefix for all endpoints.",
    )

    reload: bool = Field(
        default=_server_config.get("reload", False),
        description="Whether to enable auto-reload on code changes.",
    )

    server_header: bool = Field(
        default=_server_config.get("server_header", False),
        description="Whether to include server information in headers.",
    )

    allow_origin: list[str] = Field(
        default=_server_config.get("allow_origin", ["http://localhost:3000"]),
        description="CORS allowed origins for cross-origin requests.",
    )

    # Dataset configuration
    dataset_upload_dir: Path = Field(
        default=Path(_dataset_config.get("dataset_upload_dir")),
        description="Directory for storing uploaded dataset files.",
    )

    allowed_extensions: frozenset = Field(
        default=frozenset(_dataset_config.get("allowed_extensions", [])),
        description="File extensions permitted for dataset uploads.",
    )

    max_filename_length: int = Field(
        default=_dataset_config.get("max_filename_length"),
        description="Maximum allowed length for uploaded filenames.",
    )

    default_delimiter: str = Field(
        default=_dataset_config.get("default_delimiter"),
        description="Default delimiter used for CSV processing.",
    )

    # Model configuration
    dataset_max_upload_size: int = Field(
        default=_dataset_config.get("max_upload_size"),
        description="Maximum allowed file size for dataset uploads in bytes.",
    )

    model_max_upload_size: int = Field(
        default=_model_config.get("max_upload_size"),
        description="Maximum allowed size for model uploads in bytes.",
    )

    # Inference configuration
    inference_device: str = Field(
        default=_inference_config.get("device"),
        description="Device to use for model inference (CPU/GPU).",
    )

    # Database configuration
    db_url: str = Field(
        description="Database connection URL.", validation_alias="DATABASE_URL"
    )

    db_logging: bool = Field(
        default=_db_config.get("logging", False),
        description="Whether to enable SQL query logging.",
    )

    # Log configuration
    log_dir: str = Field(
        default=_log_config.get("log_dir"),
        description="Directory for storing log files.",
    )

    log_file: str = Field(
        default=_log_config.get("log_file"), description="Name of the log file."
    )

    rotation: str = Field(
        default=_log_config.get("rotation"),
        description="Log rotation strategy (time or size-based).",
    )

    @computed_field
    def model_upload_dir(self) -> Path:
        """Directory for storing model files."""
        if self.app_env == "testing":
            return Path("test_data/inference")
        return Path(_model_config.get("model_upload_dir"))

    @computed_field
    def log_path(self) -> Path:
        """Path where application logs are stored."""
        return Path(self.log_dir) / self.log_file

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )


# Create a single instance of Settings to use throughout the application
settings = Settings()
