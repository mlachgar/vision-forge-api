"""Minimal package entrypoint."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path

CONFIG_VERSION = "0.1.0"


def resolve_version() -> str:
    try:
        return pkg_version("vision-forge-api")
    except PackageNotFoundError:
        return CONFIG_VERSION


def create_app(config_dir: Path | str | None = None):
    from .api.app import create_app as _create_app

    return _create_app(config_dir)


__all__ = ["CONFIG_VERSION", "create_app", "resolve_version"]

__version__ = resolve_version()
