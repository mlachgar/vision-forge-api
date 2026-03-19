from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..config.schema import AuthRole


class ApiKeyEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    key_hash: str
    roles: tuple[AuthRole, ...] = Field(default_factory=tuple)
    enabled: bool = True
