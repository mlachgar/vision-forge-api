"""Helpers for precomputed embeddings on disk."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


TEXT_EMBEDDINGS_FILENAME = "text_embeddings.json"
TEXT_EMBEDDING_FORMAT_VERSION = 2


class EmbeddingStore:
    """Disk-backed storage for precomputed text embeddings."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)
        self._path = self._directory / TEXT_EMBEDDINGS_FILENAME

    def _read_payload(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        return json.loads(self._path.read_text(encoding="utf-8"))

    def load(self) -> dict[str, tuple[float, ...]]:
        raw = self._read_payload()
        vectors = raw.get("vectors", {})
        if not isinstance(vectors, dict):
            raise ValueError(
                "text_embeddings.json payload must contain a mapping under 'vectors'"
            )
        result: dict[str, tuple[float, ...]] = {}
        for key, value in vectors.items():
            if not isinstance(value, list):
                raise ValueError("Embedding vectors must be lists of floats")
            result[key] = tuple(float(v) for v in value)
        return result

    def load_metadata(self) -> dict[str, Any]:
        raw = self._read_payload()
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, dict):
            return {}
        return metadata

    def persist(
        self,
        data: Mapping[str, Iterable[float]],
        *,
        model_id: str | None = None,
        format_version: int = TEXT_EMBEDDING_FORMAT_VERSION,
    ) -> None:
        payload = {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "format_version": int(format_version),
                "model_id": model_id or "",
            },
            "vectors": {
                key: [float(v) for v in values] for key, values in data.items()
            },
        }
        temp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(self._path)
