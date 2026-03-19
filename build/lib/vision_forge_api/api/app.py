"""FastAPI application factory and shared wiring."""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path

from fastapi import FastAPI

from ..config.loader import ConfigLoader
from .context_builder import build_context
from .routers.admin import router as admin_router
from .routers.catalog import router as catalog_router
from .routers.health import router as health_router
from .routers.predict import router as predict_router


CONFIG_VERSION = "0.1.0"


def resolve_version() -> str:
    try:
        return pkg_version("vision-forge-api")
    except PackageNotFoundError:
        return CONFIG_VERSION


def create_app(config_dir: Path | str | None = None) -> FastAPI:
    loader = ConfigLoader(config_dir)
    app_version = resolve_version()
    context = build_context(loader, app_version)
    app = FastAPI(
        title=context.settings.app_name,
        version=app_version,
        description="SigLIP-powered image tagging REST API",
    )
    app.state.context = context
    app.include_router(health_router)
    app.include_router(catalog_router)
    app.include_router(predict_router)
    app.include_router(admin_router)

    logger = logging.getLogger("vision_forge_api")

    @app.on_event("startup")
    async def _log_startup() -> None:
        logger.info("Starting %s v%s", context.settings.app_name, app_version)

    return app
