"""Define configuration for the project."""

import os
import tomllib
from pathlib import Path
from typing import Final, Literal

from dotenv import load_dotenv

load_dotenv()


_app_config = Path(__file__).parent / "resources" / "app.toml"

with Path.open(_app_config, "rb") as f:
    _config = tomllib.load(f)
    _app_config = _config.get("app", {})


# Environment configuration
app_env: Literal["development", "production"] = os.getenv("ENV", "development").lower()

# Server configuration
_server_config = _app_config.get("server", {})
host_binding: Final[str] = _server_config.get("host_binding", "127.0.0.1")
port: Final[int] = _server_config.get("port", 8000)
prefix: Final[str] = _server_config.get("prefix")
reload: Final[bool] = _server_config.get("reload", False)
server_header: Final[bool] = _server_config.get("server_header", False)
allow_origin: Final[str] = _server_config.get("allow_origin", ["http://localhost:3000"])

# Dataset configuration
_dataset_config = _app_config.get("dataset", {})
dataset_upload_dir = Path(_dataset_config.get("dataset_upload_dir"))
allowed_extensions: Final[list[str]] = frozenset(
    _dataset_config.get("allowed_extensions")
)
max_upload_size: Final[int] = _dataset_config.get("max_upload_size")
max_filename_length: Final[int] = _dataset_config.get("max_filename_length")
default_delimiter: Final[str] = _dataset_config.get("default_delimiter")

# Model configuration
_model_config = _app_config.get("model", {})
model_upload_dir: Final[str] = Path(_model_config.get("model_upload_dir"))
max_upload_size: Final[int] = _model_config.get("max_upload_size")

# Inference configuration
_inference_config = _app_config.get("inference", {})
inference_device: Final[str] = _inference_config.get("device")

# Database configuration
_db_config = _app_config.get("db", {})
db_url: Final[str] = os.getenv("DATABASE_URL")
db_logging: Final[bool] = _db_config.get("logging", False)

# Log configuration
_log_config = _app_config.get("logging", {})
log_path: Final[str] = Path(_log_config.get("log_dir")) / _log_config.get("log_file")
rotation: Final[str] = _log_config.get("rotation")
