"""Authentication cache and token utilities."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..config.schema import AuthConfig, AuthRole
from .models import ApiKeyEntry

DEFAULT_DATA_DIR = Path("/data")
ENV_DATA_DIR = "VISION_FORGE_DATA_DIR"
API_KEYS_FILENAME = "api_keys.json"
TokenHash = str


class AuthError(Exception):
    """Raised when a token cannot be validated."""


class AuthTokenManager:
    """Helpers for generating and hashing bearer tokens."""

    def __init__(self, config: AuthConfig):
        self._config = config

    def generate_token(self) -> str:
        suffix_len = max(self._config.token_length - len(self._config.token_prefix), 1)
        random_segment = secrets.token_urlsafe(suffix_len)
        token = f"{self._config.token_prefix}{random_segment}"
        return token[: self._config.token_length]

    def hash_token(self, token: str) -> TokenHash:
        return hash_token(token)


def hash_token(token: str) -> TokenHash:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def parse_authorization_header(value: str | None) -> str | None:
    if not value:
        return None
    parts = value.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise AuthError("Authorization header must be 'Bearer <token>'")
    return parts[1]


def _resolve_data_dir(path: Path | str | None = None) -> Path:
    candidate = path or os.getenv(ENV_DATA_DIR) or DEFAULT_DATA_DIR
    return Path(candidate).expanduser().resolve(strict=False)


def _read_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("api_keys.json must contain a list of entries")
    return raw


class ApiKeyRepository:
    """Disk-backed storage for persisted API keys."""

    def __init__(self, data_dir: Path | str | None = None):
        self._data_dir = _resolve_data_dir(data_dir)
        self._path = self._data_dir / API_KEYS_FILENAME

    @property
    def path(self) -> Path:
        return self._path

    def read_all(self) -> list[ApiKeyEntry]:
        return [ApiKeyEntry.model_validate(entry) for entry in _read_entries(self._path)]

    def persist(self, entries: Iterable[ApiKeyEntry]) -> None:
        payload = [entry.model_dump() for entry in entries]
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(self._path)


class AuthCache:
    """In-memory cache for fast token validation."""

    def __init__(self, entries: Sequence[ApiKeyEntry]):
        self._entries: dict[TokenHash, ApiKeyEntry] = {}
        self.reload(entries)

    @classmethod
    def from_repository(cls, repo: ApiKeyRepository) -> "AuthCache":
        return cls(repo.read_all())

    @property
    def entries(self) -> tuple[ApiKeyEntry, ...]:
        return tuple(self._entries.values())

    def reload(self, entries: Sequence[ApiKeyEntry]) -> None:
        self._entries = {entry.key_hash: entry for entry in entries}

    def lookup(self, key_hash: TokenHash) -> ApiKeyEntry | None:
        return self._entries.get(key_hash)

    def authorize(self, token: str, required_role: AuthRole | None = None) -> ApiKeyEntry:
        key_hash = hash_token(token)
        entry = self.lookup(key_hash)
        if entry is None:
            raise AuthError("API key not found")
        if not entry.enabled:
            raise AuthError("API key is disabled")
        if required_role and required_role not in entry.roles:
            raise AuthError("API key lacks the required role")
        return entry
