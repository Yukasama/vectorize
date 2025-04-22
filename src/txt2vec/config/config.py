"""Define configuration for the project."""

import tomllib
from pathlib import Path

__all__ = ["app_config"]

app_config = Path(__file__).parent / "resources" / "app.toml"

with open(app_config, "rb") as f:
    _config = tomllib.load(f)
    app_config = _config.get("app", {})
