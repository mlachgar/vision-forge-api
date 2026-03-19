#!/usr/bin/env python3
"""Download and cache the SigLIP model weights indicated in settings."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vision_forge_api.config.loader import ConfigLoader
from vision_forge_api.siglip.service import SiglipService

LOGGER = logging.getLogger("download_model")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ensure the SigLIP model weights are cached locally.")
    parser.add_argument("--config-dir", type=Path, default=Path("/config"), help="Mounted /config path")
    parser.add_argument("--device", default=None, help="Preferred device (cpu, cuda, etc.)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    loader = ConfigLoader(args.config_dir)
    settings = loader.load_settings()
    SiglipService(settings.siglip_model_id, settings.model_cache_dir, device_hint=args.device)
    LOGGER.info("Model %s is available in %s", settings.siglip_model_id, settings.model_cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
