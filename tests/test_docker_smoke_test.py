from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts import docker_smoke_test as smoke


def test_prepare_runtime_data_seeds_demo_api_keys(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    args = SimpleNamespace(
        data_dir=data_dir,
        predict_token=smoke.DEFAULT_PREDICT_TOKEN,
    )

    smoke._prepare_runtime_data(args)

    api_keys_path = data_dir / "api_keys.json"
    assert api_keys_path.is_file()

    payload = json.loads(api_keys_path.read_text(encoding="utf-8"))
    assert [entry["name"] for entry in payload] == ["admin", "predictor"]
    assert payload[0]["key_hash"] == smoke._hash_token(smoke.DEFAULT_ADMIN_TOKEN)
    assert payload[0]["roles"] == ["admin", "predict"]
    assert payload[1]["key_hash"] == smoke._hash_token(smoke.DEFAULT_PREDICT_TOKEN)
    assert payload[1]["roles"] == ["predict"]
    assert (data_dir / "embeddings").is_dir()
    assert (data_dir / "model_cache").is_dir()


def test_prepare_runtime_data_keeps_existing_api_keys(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    existing = data_dir / "api_keys.json"
    existing.write_text("[]", encoding="utf-8")
    args = SimpleNamespace(
        data_dir=data_dir,
        predict_token="other-token",
    )

    smoke._prepare_runtime_data(args)

    assert existing.read_text(encoding="utf-8") == "[]"
