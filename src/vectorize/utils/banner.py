"""Banner generation for application."""

import platform
import sys
from datetime import UTC, datetime

from pyfiglet import figlet_format

from vectorize.config.config import Settings
from vectorize.utils.file_size_formatter import format_file_size


def create_banner(settings: Settings, silent: bool = False) -> str:
    """Generate and optionally print a banner with server name and settings.

    Args:
        settings: Application configuration settings
        silent: If True, suppress console output and return banner as string

    Returns:
        The complete banner as a string
    """
    lines: list[str] = []

    banner = figlet_format("VECTORIZE", font="slant")
    lines.extend([
        "\033[1;36m" + banner + "\033[0m",
        "\033[1;33m⚡ VECTORIZE Service v0.1.0 ⚡\033[0m",
        f"\033[0;37m{'-' * 60}\033[0m",
    ])

    env_color = "\033[1;31m" if settings.app_env == "production" else "\033[1;32m"
    lines.extend([
        f"🌍 Environment: {env_color}{settings.app_env}\033[0m",
        f"🔌 API: http://{'0.0.0.0' if settings.host_binding == '0.0.0.0' else 'localhost'}:{settings.port}{settings.prefix}",  # noqa: E501, S104
        f"📋 Docs: http://localhost:{settings.port}/docs",
        f"📊 Metrics: http://localhost:{settings.port}/metrics",
    ])

    lines.extend([
        "\n\033[1;33m💾 Database Configuration\033[0m",
        f"  • Engine: {settings.db_url.split('://')[0]}",
        f"  • Pool Size: {settings.db_pool_size} (max: {settings.db_pool_size + settings.db_max_overflow})",  # noqa: E501
        f"  • Clear on Restart: {'✅' if settings.clear_db_on_restart else '❌'}",
    ])

    lines.extend([
        "\n\033[1;33m📁 Storage Configuration\033[0m",
        f"  • Dataset Directory: {settings.dataset_upload_dir}",
        f"  • Model Directory: {settings.model_upload_dir}",
        f"  • Max Dataset Size: {format_file_size(settings.dataset_max_upload_size)}",
        f"  • Max Model Size: {format_file_size(settings.model_max_upload_size)}",
    ])

    lines.extend([
        "\n\033[1;33m🧠 Inference Configuration\033[0m",
        f"  • Device: {settings.inference_device.upper()}",
    ])

    lines.extend([
        "\n\033[1;33m📝 Logging Configuration\033[0m",
        f"  • Log Level: {settings.log_level}",
        f"  • Log Path: {settings.log_path}",
    ])

    lines.extend([
        "\n\033[1;33m⚙️ System Information\033[0m",
        f"  • OS: {platform.system()} {platform.release()}",
        f"  • Python: {sys.version.split()[0]}",
        f"  • Auto Reload: {'✅' if settings.reload else '❌'}",
        f"  • Started at: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"\033[0;37m{'-' * 60}\033[0m",
    ])

    banner_text = "\n".join(lines)
    if not silent:
        print(banner_text)  # noqa: T201
    return banner_text
