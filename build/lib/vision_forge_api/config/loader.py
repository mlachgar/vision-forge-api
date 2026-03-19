from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .schema import (
    AuthConfig,
    ProfilesConfig,
    PromptsConfig,
    SettingsConfig,
    TagSetsConfig,
)


DEFAULT_CONFIG_DIR = Path("/config")
ENV_CONFIG_DIR = "VISION_FORGE_CONFIG_DIR"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Expected config at {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config {path} must be a mapping")
    return raw


class ConfigLoader:
    def __init__(self, config_dir: Path | str | None = None):
        candidate = config_dir or os.getenv(ENV_CONFIG_DIR) or DEFAULT_CONFIG_DIR
        self.config_dir = Path(candidate).expanduser().resolve()
        if not self.config_dir.is_dir():
            raise FileNotFoundError(f"Config directory {self.config_dir} is missing")

    def load_auth(self) -> AuthConfig:
        return AuthConfig.model_validate(_read_yaml(self.config_dir / "auth.yaml"))

    def load_settings(self) -> SettingsConfig:
        return SettingsConfig.model_validate(_read_yaml(self.config_dir / "settings.yaml"))

    def load_tag_sets(self) -> TagSetsConfig:
        return TagSetsConfig.model_validate(_read_yaml(self.config_dir / "tag_sets.yaml"))

    def load_profiles(self) -> ProfilesConfig:
        return ProfilesConfig.model_validate(_read_yaml(self.config_dir / "profiles.yaml"))

    def load_prompts(self) -> PromptsConfig:
        return PromptsConfig.model_validate(_read_yaml(self.config_dir / "prompts.yaml"))
