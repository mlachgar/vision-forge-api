from __future__ import annotations

from types import SimpleNamespace

import pytest

from vision_forge_api.api.errors import ApiKeyNotFoundError, ConflictError
from vision_forge_api.api.services.admin import AdminService
from vision_forge_api.auth.models import ApiKeyEntry


class _Repo:
    def __init__(self, entries: list[ApiKeyEntry]) -> None:
        self._entries = list(entries)

    def read_all(self) -> list[ApiKeyEntry]:
        return list(self._entries)

    def persist(self, entries: list[ApiKeyEntry]) -> None:
        self._entries = list(entries)


class _Cache:
    def __init__(self, entries: list[ApiKeyEntry]) -> None:
        self._entries = tuple(entries)

    @property
    def entries(self) -> tuple[ApiKeyEntry, ...]:
        return self._entries

    def reload(self, entries: list[ApiKeyEntry]) -> None:
        self._entries = tuple(entries)


class _TokenManager:
    def generate_token(self) -> str:
        return "token-value"

    def hash_token(self, token: str) -> str:
        return f"sha256:{token}"


def _context(entries: list[ApiKeyEntry]) -> SimpleNamespace:
    return SimpleNamespace(
        api_key_repo=_Repo(entries),
        auth_cache=_Cache(entries),
        token_manager=_TokenManager(),
        auth_config=SimpleNamespace(default_roles=("predict",)),
        loader=SimpleNamespace(),
        version="0.1.0",
    )


def _entry(name: str) -> ApiKeyEntry:
    return ApiKeyEntry(
        name=name, key_hash=f"sha256:{name}", roles=("predict",), enabled=True
    )


def test_update_missing_api_key_raises_centralized_not_found_error() -> None:
    service = AdminService(_context([_entry("existing")]))

    with pytest.raises(ApiKeyNotFoundError) as exc:
        service.update_api_key_enabled("missing", enabled=False)

    assert exc.value.detail == "API key not found"


def test_delete_missing_api_key_raises_centralized_not_found_error() -> None:
    service = AdminService(_context([_entry("existing")]))

    with pytest.raises(ApiKeyNotFoundError):
        service.delete_api_key("missing")


def test_create_api_key_duplicate_name_raises_conflict() -> None:
    service = AdminService(_context([_entry("dup")]))

    with pytest.raises(ConflictError):
        service.create_api_key(name="dup", roles=None, enabled=None)


def test_create_api_key_uses_defaults_and_reload_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context([_entry("existing")])
    service = AdminService(context)
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        "vision_forge_api.api.services.admin.build_context",
        lambda loader, version: seen.setdefault(
            "context", SimpleNamespace(loader=loader, version=version)
        ),
    )

    created = service.create_api_key(name="new-key", roles=None, enabled=None)

    assert created.token == "token-value"
    assert created.roles == ("predict",)
    assert created.enabled is True
    assert [entry.name for entry in context.api_key_repo.read_all()] == [
        "existing",
        "new-key",
    ]
    assert [entry.name for entry in context.auth_cache.entries] == [
        "existing",
        "new-key",
    ]

    reloaded = service.reload_configuration()
    assert reloaded.loader is context.loader
    assert reloaded.version == context.version
