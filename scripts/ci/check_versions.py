#!/usr/bin/env python3
"""Validate that a release Git tag matches pyproject version."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import tomllib

SEMVER_TAG_PATTERN = re.compile(r"^v(?P<version>\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?)$")


def _read_project_version(pyproject_path: Path) -> str:
    document = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = document.get("project")
    if not isinstance(project, dict):
        raise ValueError("[project] section is missing in pyproject.toml")
    version = project.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("project.version is missing in pyproject.toml")
    return version.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Ensure release tag matches pyproject version.")
    parser.add_argument("--tag", required=True, help="Git tag, for example v1.2.3")
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    args = parser.parse_args()

    match = SEMVER_TAG_PATTERN.match(args.tag)
    if not match:
        raise SystemExit(f"Invalid tag '{args.tag}'. Expected format: v1.2.3")

    tag_version = match.group("version")
    project_version = _read_project_version(args.pyproject)
    if project_version != tag_version:
        raise SystemExit(
            "Release tag version does not match pyproject version: "
            f"tag={tag_version}, pyproject={project_version}"
        )

    print(f"Version check passed: {tag_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
