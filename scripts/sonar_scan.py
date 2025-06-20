"""Run SonarQube scanner for code analysis."""

import os
import platform
import subprocess  # noqa: S404
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sonar_token = os.getenv("SONAR_TOKEN")

_SONAR_BAT = "sonar-scanner.bat"


def run_sonar_scan() -> None:
    """Run SonarQube scanner for code analysis."""
    if sonar_token is None:
        print("❌ SONAR_TOKEN is not set. Please set it in your environment variables.")  # noqa: T201
        sys.exit(1)

    try:
        if platform.system() == "Windows":
            scoop_path = (
                Path.home() / "scoop" / "apps" / "sonar-scanner" / "current" / "bin"
            )
            executable_path = scoop_path / _SONAR_BAT

            if executable_path.exists():
                scanner_cmd = str(executable_path)
            else:
                sonar_home = os.getenv("SONAR_SCANNER_HOME")
                if sonar_home and Path(sonar_home).exists():
                    scanner_cmd = str(Path(sonar_home) / "bin" / _SONAR_BAT)
                else:
                    scanner_cmd = _SONAR_BAT
        else:
            scanner_cmd = "sonar-scanner"

        subprocess.run([scanner_cmd, f"-Dsonar.login={sonar_token}"], check=True)  # noqa: S603
        print("✅ Sonar scan completed successfully.")  # noqa: T201
    except subprocess.CalledProcessError as e:
        print("❌ Sonar scan failed.", file=sys.stderr)  # noqa: T201
        sys.exit(e.returncode)
    except FileNotFoundError:
        print(  # noqa: T201
            "❌ Sonar scanner not found. Check Scoop or PATH.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    run_sonar_scan()
