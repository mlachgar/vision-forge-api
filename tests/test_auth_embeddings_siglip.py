from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vision_forge_api.auth.cache import (
    ApiKeyRepository,
    AuthCache,
    AuthError,
    AuthTokenManager,
    hash_token,
    parse_authorization_header,
)
from vision_forge_api.auth.models import ApiKeyEntry
from vision_forge_api.config.schema import AuthConfig, AuthRole
from vision_forge_api.embeddings.store import EmbeddingStore
from vision_forge_api.siglip import service as siglip_mod


def _entry(
    name: str, token: str, roles: tuple[AuthRole, ...], enabled: bool = True
) -> ApiKeyEntry:
    return ApiKeyEntry(
        name=name, key_hash=hash_token(token), roles=roles, enabled=enabled
    )


def test_auth_token_manager_and_parser() -> None:
    manager = AuthTokenManager(AuthConfig(token_prefix="vfk_", token_length=16))
    token = manager.generate_token()
    assert token.startswith("vfk_")
    assert len(token) == 16

    hashed = manager.hash_token(token)
    assert hashed.startswith("sha256:")
    assert parse_authorization_header(f"Bearer {token}") == token
    assert parse_authorization_header(None) is None
    with pytest.raises(AuthError):
        parse_authorization_header("invalid")


def test_api_key_repository_and_auth_cache(tmp_path: Path) -> None:
    repo = ApiKeyRepository(data_dir=tmp_path)
    assert repo.read_all() == []

    enabled = _entry("enabled", "token-1", (AuthRole.PREDICT,), enabled=True)
    disabled = _entry("disabled", "token-2", (AuthRole.PREDICT,), enabled=False)
    repo.persist([enabled, disabled])

    loaded = repo.read_all()
    assert len(loaded) == 2
    cache = AuthCache.from_repository(repo)

    missing = cache.authorize("missing", required_role=AuthRole.PREDICT)
    assert missing.status_code == 401

    off = cache.authorize("token-2", required_role=AuthRole.PREDICT)
    assert off.status_code == 401

    forbidden = cache.authorize("token-1", required_role=AuthRole.ADMIN)
    assert forbidden.status_code == 403

    ok = cache.authorize("token-1", required_role=AuthRole.PREDICT)
    assert ok.status_code == 200
    assert ok.entry is not None


def test_embedding_store_roundtrip_and_validation(tmp_path: Path) -> None:
    store = EmbeddingStore(tmp_path)
    assert store.load() == {}
    assert store.load_metadata() == {}

    store.persist({"cat": [0.1, 0.2]}, model_id="m1", format_version=7)
    assert store.load()["cat"] == (0.1, 0.2)
    metadata = store.load_metadata()
    assert metadata["model_id"] == "m1"
    assert metadata["format_version"] == 7

    raw = json.loads((tmp_path / "text_embeddings.json").read_text(encoding="utf-8"))
    raw["vectors"] = []
    (tmp_path / "text_embeddings.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        store.load()


def test_siglip_service_behaviors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(siglip_mod.torch.cuda, "is_available", lambda: False)
    assert str(siglip_mod._resolve_device("gpu")) == "cpu"

    class _Processor:
        def __call__(
            self,
            *,
            images=None,
            text=None,
            return_tensors=None,
            padding=None,
            truncation=None,
        ):
            if images is not None:
                return SimpleNamespace(
                    pixel_values=torch.tensor([[1.0]], dtype=torch.float32)
                )
            return {
                "input_ids": torch.tensor([[1.0]], dtype=torch.float32),
                "attention_mask": torch.tensor([[1.0]], dtype=torch.float32),
            }

    class _Model:
        def to(self, _device):
            return self

        def eval(self):
            return None

        def get_image_features(self, *, pixel_values):
            return torch.tensor([[3.0, 4.0]], dtype=torch.float32)

        def get_text_features(self, **_kwargs):
            return torch.tensor([[6.0, 8.0]], dtype=torch.float32)

    monkeypatch.setattr(
        siglip_mod.SiglipProcessor,
        "from_pretrained",
        lambda *args, **kwargs: _Processor(),
    )
    monkeypatch.setattr(
        siglip_mod.SiglipModel, "from_pretrained", lambda *args, **kwargs: _Model()
    )

    service = siglip_mod.SiglipService("model-id", tmp_path, device_hint="cpu")

    image_vec = service.encode_image(image=SimpleNamespace())
    assert image_vec.shape[-1] == 2
    assert torch.allclose(
        torch.linalg.vector_norm(image_vec, dim=-1), torch.ones(1), atol=1e-5
    )

    text_vec = service.encode_texts(("cat",))
    assert text_vec.shape[-1] == 2

    empty = service.encode_texts(())
    assert empty.numel() == 0
