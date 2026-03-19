from __future__ import annotations

import os

import uvicorn

from .api.app import create_app


def main() -> None:
    """Entrypoint for console scripts."""
    host = os.getenv("VISION_FORGE_HOST", "0.0.0.0")
    port = int(os.getenv("VISION_FORGE_PORT", "8000"))
    config_dir = os.getenv("VISION_FORGE_CONFIG_DIR")
    app = create_app(config_dir)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
