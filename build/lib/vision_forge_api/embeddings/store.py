"""Helpers for precomputed embeddings on disk."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping


TEXT_EMBEDDINGS_FILENAME = "text_embeddings.json"


class EmbeddingStore:
    """Disk-backed storage for precomputed text embeddings."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)
        self._path = self._directory / TEXT_EMBEDDINGS_FILENAME

    def load(self) -> dict[str, tuple[float, ...]]:
        if not self._path.exists():
            return {}
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        vectors = raw.get("vectors", {})
        if not isinstance(vectors, dict):
            raise ValueError("text_embeddings.json payload must contain a mapping under 'vectors'")
        result: dict[str, tuple[float, ...]] = {}
        for key, value in vectors.items():
            if not isinstance(value, list):
                raise ValueError("Embedding vectors must be lists of floats")
            result[key] = tuple(float(v) for v in value)
        return result

    def persist(self, data: Mapping[str, Iterable[float]]) -> None:
        payload = {
            "version": 1,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "vectors": {key: [float(v) for v in values] for key, values in data.items()},
        }
        temp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(self._path)
