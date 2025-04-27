#!/usr/bin/env python

"""Fail CI when a production dependency is outdated (uv-only version)."""

import json
import subprocess  # noqa: S404
import sys
import tomllib
from pathlib import Path
from typing import Any

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def load_prod_deps(pyproject: Path) -> set[str]:
    """Return the set of package names declared in [project.dependencies]."""
    try:
        with pyproject.open("rb") as fh:
            data: dict[str, Any] = tomllib.load(fh)
        return {
            dep.split()[0].split("==")[0].split(">")[0].split("<")[0].lower()
            for dep in data["project"]["dependencies"]
        }
    except (FileNotFoundError, KeyError, tomllib.TOMLDecodeError):
        print("::error ::Cannot read production dependencies from pyproject.toml")
        sys.exit(2)


def uv_outdated() -> list[dict[str, str]]:
    """Call `uv pip list --outdated --format=json` and return parsed JSON."""
    try:
        cp = subprocess.run(
            ["uv", "pip", "list", "--outdated", "--format=json"],  # noqa: S607
            check=True,
            text=True,
            capture_output=True,
        )
        return json.loads(cp.stdout or "[]")
    except FileNotFoundError:
        print("::error ::`uv` executable not found on PATH")
        sys.exit(2)
    except subprocess.CalledProcessError as exc:
        print(f"::error ::uv pip list failed: {exc.stderr.strip()}")
        sys.exit(exc.returncode)


def main() -> None:
    """Main function to check for outdated production dependencies."""
    prod = load_prod_deps(PYPROJECT)
    outdated_prod = [p for p in uv_outdated() if p["name"].lower() in prod]

    if not outdated_prod:
        sys.exit(0)

    print("Outdated production dependencies:")
    for pkg in sorted(outdated_prod, key=lambda d: d["name"].lower()):
        print(f"- {pkg['name']} {pkg['version']} → {pkg['latest_version']}")
    sys.exit(1)


if __name__ == "__main__":
    main()
