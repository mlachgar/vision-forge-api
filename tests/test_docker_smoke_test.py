from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

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
    assert not hasattr(args, "config_dir") or not getattr(args, "config_dir", None)


def test_prepare_runtime_data_creates_config_dir_when_requested(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    args = SimpleNamespace(
        data_dir=data_dir,
        config_dir=config_dir,
        predict_token=smoke.DEFAULT_PREDICT_TOKEN,
    )

    smoke._prepare_runtime_data(args)

    assert config_dir.is_dir()
    assert (config_dir.stat().st_mode & 0o777) == 0o755


def test_start_container_mounts_full_data_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    args = SimpleNamespace(
        data_dir=data_dir,
        image="example.com/vision-forge-api:cpu-full",
    )
    captured: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool = True):
        captured.append(command)
        return SimpleNamespace(returncode=0, stdout="container-123\n", stderr="")

    monkeypatch.setattr(smoke, "_run", fake_run)

    container_id = smoke._start_container(args)

    assert container_id == "container-123"
    assert captured[0][0:3] == ["docker", "run", "-d"]
    assert f"{data_dir.resolve()}:/data" in captured[0]


def test_start_container_mounts_config_dir_when_requested(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    data_dir.mkdir()
    config_dir.mkdir()
    args = SimpleNamespace(
        data_dir=data_dir,
        config_dir=config_dir,
        image="example.com/vision-forge-api:cpu-full",
    )
    captured: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool = True):
        captured.append(command)
        return SimpleNamespace(returncode=0, stdout="container-123\n", stderr="")

    monkeypatch.setattr(smoke, "_run", fake_run)

    container_id = smoke._start_container(args)

    assert container_id == "container-123"
    assert f"{data_dir.resolve()}:/data" in captured[0]
    assert f"{config_dir.resolve()}:/config:ro" in captured[0]
    assert "VISION_FORGE_CONFIG_DIR=/config" in captured[0]
