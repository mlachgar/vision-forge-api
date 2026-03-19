"""Business logic for admin endpoints."""

from __future__ import annotations

from dataclasses import dataclass

from ...auth.models import ApiKeyEntry
from ...config.schema import AuthRole
from ..context import AppContext
from ..context_builder import build_context
from ..errors import ApiKeyNotFoundError, ConflictError


@dataclass(frozen=True)
class CreatedApiKey:
    name: str
    token: str
    roles: tuple[AuthRole, ...]
    enabled: bool


class AdminService:
    """Coordinates API key persistence and app reload operations."""

    def __init__(self, context: AppContext) -> None:
        self._context = context

    def list_api_keys(self) -> tuple[ApiKeyEntry, ...]:
        return self._context.auth_cache.entries

    def create_api_key(
        self,
        name: str,
        roles: tuple[AuthRole, ...] | None,
        enabled: bool | None,
    ) -> CreatedApiKey:
        entries = self._context.api_key_repo.read_all()
        if any(entry.name == name for entry in entries):
            raise ConflictError("API key with that name already exists")

        effective_roles = (
            roles
            if roles is not None
            else tuple(self._context.auth_config.default_roles)
        )
        effective_enabled = enabled if enabled is not None else True

        token = self._context.token_manager.generate_token()
        key_hash = self._context.token_manager.hash_token(token)

        created = ApiKeyEntry(
            name=name,
            key_hash=key_hash,
            roles=effective_roles,
            enabled=effective_enabled,
        )
        entries.append(created)
        self._persist_entries(entries)

        return CreatedApiKey(
            name=created.name,
            token=token,
            roles=created.roles,
            enabled=created.enabled,
        )

    def update_api_key_enabled(self, name: str, enabled: bool | None) -> ApiKeyEntry:
        entries = self._context.api_key_repo.read_all()
        idx = self._find_entry_index(entries, name)
        updated = (
            entries[idx]
            if enabled is None
            else entries[idx].model_copy(update={"enabled": enabled})
        )
        entries[idx] = updated
        self._persist_entries(entries)
        return updated

    def delete_api_key(self, name: str) -> None:
        entries = self._context.api_key_repo.read_all()
        idx = self._find_entry_index(entries, name)
        del entries[idx]
        self._persist_entries(entries)

    def reload_configuration(self) -> AppContext:
        return build_context(self._context.loader, self._context.version)

    def _persist_entries(self, entries: list[ApiKeyEntry]) -> None:
        self._context.api_key_repo.persist(entries)
        self._context.auth_cache.reload(entries)

    @staticmethod
    def _find_entry_index(entries: list[ApiKeyEntry], name: str) -> int:
        for idx, entry in enumerate(entries):
            if entry.name == name:
                return idx
        raise ApiKeyNotFoundError()
