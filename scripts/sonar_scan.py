"""Run SonarQube scanner for code analysis."""

import os
import subprocess  # noqa: S404
import sys

from dotenv import load_dotenv

load_dotenv()

sonar_token = os.getenv("SONAR_TOKEN")


def run_sonar_scan() -> None:
    """Run SonarQube scanner for code analysis."""
    if sonar_token is None:
        print("❌ SONAR_TOKEN is not set. Please set it in your environment variables.")  # noqa: T201
        sys.exit(1)

    try:
        subprocess.run(["sonar-scanner", f"-D sonar.login=${sonar_token}"], check=True)  # noqa: S603, S607
        print("✅ Sonar scan completed successfully.")  # noqa: T201
    except subprocess.CalledProcessError as e:
        print("❌ Sonar scan failed.", file=sys.stderr)  # noqa: T201
        sys.exit(e.returncode)


if __name__ == "__main__":
    run_sonar_scan()
