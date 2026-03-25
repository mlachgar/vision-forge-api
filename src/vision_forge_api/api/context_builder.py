"""Context builder shared between app factory and admin reload."""

from __future__ import annotations

import os

from ..auth.cache import ApiKeyRepository, AuthCache, AuthTokenManager
from ..catalog.service import TagCatalog
from ..config.loader import ConfigLoader
from ..predict.service import PredictionService
from ..siglip.service import SiglipService
from .context import AppContext
from .services.predict_jobs import PredictJobService


def build_context(loader: ConfigLoader, app_version: str) -> AppContext:
    settings = loader.load_settings()
    tag_catalog = TagCatalog(
        loader.load_tag_sets(), loader.load_profiles(), loader.load_prompts()
    )
    auth_config = loader.load_auth()
    token_manager = AuthTokenManager(auth_config)
    api_key_repo = ApiKeyRepository()
    auth_cache = AuthCache.from_repository(api_key_repo)
    device_hint = os.getenv("VISION_FORGE_DEVICE")
    siglip = SiglipService(
        settings.siglip_model_id, settings.model_cache_dir, device_hint=device_hint
    )
    prediction_service = PredictionService(tag_catalog, siglip, settings.embeddings_dir)
    context = AppContext(
        loader=loader,
        settings=settings,
        auth_config=auth_config,
        version=app_version,
        config_dir=loader.config_dir,
        tag_catalog=tag_catalog,
        auth_cache=auth_cache,
        token_manager=token_manager,
        api_key_repo=api_key_repo,
        siglip_service=siglip,
        prediction_service=prediction_service,
    )
    prediction_job_service = PredictJobService(context)
    object.__setattr__(context, "prediction_job_service", prediction_job_service)
    return context
