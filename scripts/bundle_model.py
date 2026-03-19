#!/usr/bin/env python3
"""Download and cache a SigLIP model for image bundling."""

from __future__ import annotations

import argparse
from pathlib import Path

from vision_forge_api.siglip.service import SiglipService


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache SigLIP model artifacts.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    SiglipService(args.model_id, args.model_cache_dir, device_hint=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
