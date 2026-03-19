"""Shared application context."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vision_forge_api.auth.cache import ApiKeyRepository, AuthCache, AuthTokenManager
from ..catalog.service import TagCatalog
from ..config.loader import ConfigLoader
from ..config.schema import AuthConfig, SettingsConfig
from ..predict.service import PredictionService
from ..siglip.service import SiglipService


@dataclass(frozen=True)
class AppContext:
    loader: ConfigLoader
    settings: SettingsConfig
    auth_config: AuthConfig
    version: str
    tag_catalog: TagCatalog
    auth_cache: AuthCache
    token_manager: AuthTokenManager
    api_key_repo: ApiKeyRepository
    siglip_service: SiglipService
    prediction_service: PredictionService
    config_dir: Path | None = None
