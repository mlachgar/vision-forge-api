from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, Field
from pydantic import ConfigDict


class AuthRole(str, Enum):
    ADMIN = "admin"
    PREDICT = "predict"


class AuthConfig(BaseModel):
    token_prefix: str = Field("vfk_", min_length=3)
    token_length: int = Field(32, ge=16, le=64)
    admin_roles: Sequence[AuthRole] = Field(default_factory=lambda: [AuthRole.ADMIN])
    default_roles: Sequence[AuthRole] = Field(
        default_factory=lambda: [AuthRole.PREDICT]
    )


class SettingsConfig(BaseModel):
    app_name: str = Field("vision-forge-api")
    default_limit: int = Field(20, ge=1)
    max_limit: int = Field(200, ge=1)
    default_min_score: float = Field(-0.05, ge=-1.0, le=1.0)
    embeddings_dir: Path = Field(Path("/data/embeddings"))
    model_cache_dir: Path = Field(Path("/data/model_cache"))
    siglip_model_id: str = Field("google/siglip-base-patch16-224")
    model_config = ConfigDict(protected_namespaces=())


class TagPrompt(BaseModel):
    template: str
    weight: float = Field(1.0, ge=0.0)


class CanonicalTag(BaseModel):
    key: str
    label: str
    description: str | None = None
    prompts: Sequence[TagPrompt] = Field(default_factory=list)


class TagSet(BaseModel):
    name: str
    description: str | None = None
    canonical_tags: Sequence[str]


class TagSetsConfig(BaseModel):
    tag_sets: Sequence[TagSet] = Field(default_factory=list)


class Profile(BaseModel):
    name: str
    tag_sets: Sequence[str] = Field(default_factory=list)
    description: str | None = None


class ProfilesConfig(BaseModel):
    profiles: Sequence[Profile] = Field(default_factory=list)


class PromptEntry(BaseModel):
    canonical_tag: str
    prompts: Sequence[TagPrompt] = Field(default_factory=list)


class PromptsConfig(BaseModel):
    prompts: Sequence[PromptEntry] = Field(default_factory=list)
