#!/usr/bin/env python3
# ruff: noqa: E402
"""Validate mounted configuration files and supporting data directories."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vision_forge_api.auth.cache import ApiKeyRepository
from vision_forge_api.catalog.service import TagCatalog
from vision_forge_api.config.loader import ConfigLoader

LOGGER = logging.getLogger("validate_config")


def _resolve_runtime_data_path(path: Path, data_dir: Path) -> Path:
    if path.is_absolute() and path.parts[:2] == ("/", "data"):
        return data_dir / path.relative_to("/data")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ensure config files and directories meet expectations."
    )
    parser.add_argument(
        "--config-dir", type=Path, default=Path("/config"), help="Mounted /config path"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("/data"), help="Mounted /data path"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    loader = ConfigLoader(args.config_dir)
    settings = loader.load_settings()
    TagCatalog(loader.load_tag_sets(), loader.load_profiles(), loader.load_prompts())

    data_dir = args.data_dir
    api_repo = ApiKeyRepository(data_dir=data_dir)
    entries = api_repo.read_all()
    if not entries:
        LOGGER.warning("No API keys found in %s", api_repo.path)
    elif not any(entry.enabled for entry in entries):
        LOGGER.warning("All API keys are disabled; enable at least one entry")
    LOGGER.info("Found %d API key entries", len(entries))

    for configured_path in (settings.embeddings_dir, settings.model_cache_dir):
        runtime_path = _resolve_runtime_data_path(configured_path, data_dir)
        if not runtime_path.exists():
            LOGGER.info(
                "Creating missing data directory %s (configured as %s)",
                runtime_path,
                configured_path,
            )
            runtime_path.mkdir(parents=True, exist_ok=True)
        if not runtime_path.is_dir():
            LOGGER.error("Expected %s to be a directory", runtime_path)
            return 1
    LOGGER.info("Config validation succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
