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


def test_api_key_repository_env_resolution_and_invalid_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VISION_FORGE_DATA_DIR", str(data_dir))

    repo = ApiKeyRepository()
    assert repo.path == data_dir / "api_keys.json"
    assert repo.read_all() == []

    repo.path.parent.mkdir(parents=True, exist_ok=True)
    repo.path.write_text("null", encoding="utf-8")
    assert repo.read_all() == []

    repo.path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        repo.read_all()

    entry = _entry("enabled", "token-1", (AuthRole.PREDICT,), enabled=True)
    cache = AuthCache([entry])
    assert cache.entries == (entry,)
    cache.reload([])
    assert cache.entries == ()


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


def test_embedding_store_handles_non_mapping_metadata(tmp_path: Path) -> None:
    store = EmbeddingStore(tmp_path)
    payload = {
        "version": 1,
        "metadata": [],
        "vectors": {"cat": [0.1]},
    }
    (tmp_path / "text_embeddings.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    assert store.load_metadata() == {}


def test_embedding_store_rejects_non_list_vectors(tmp_path: Path) -> None:
    store = EmbeddingStore(tmp_path)
    payload = {
        "version": 1,
        "metadata": {},
        "vectors": {"cat": 0.1},
    }
    (tmp_path / "text_embeddings.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    with pytest.raises(ValueError):
        store.load()


def test_siglip_service_behaviors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(siglip_mod.torch.cuda, "is_available", lambda: False)
    assert str(siglip_mod._resolve_device("gpu")) == "cpu"

    load_calls: list[tuple[str, tuple, dict]] = []

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
            return SimpleNamespace(
                pooler_output=torch.tensor([[3.0, 4.0]], dtype=torch.float32)
            )

        def get_text_features(self, **_kwargs):
            return SimpleNamespace(
                pooler_output=torch.tensor([[6.0, 8.0]], dtype=torch.float32)
            )

    monkeypatch.setattr(
        siglip_mod.SiglipProcessor,
        "from_pretrained",
        lambda *args, **kwargs: (
            load_calls.append(("processor", args, kwargs)) or _Processor()
        ),
    )
    monkeypatch.setattr(
        siglip_mod.SiglipModel,
        "from_pretrained",
        lambda *args, **kwargs: load_calls.append(("model", args, kwargs)) or _Model(),
    )

    service = siglip_mod.SiglipService("model-id", tmp_path, device_hint="cpu")
    assert [call[0] for call in load_calls] == ["processor", "model"]

    image_vec = service.encode_image(image=SimpleNamespace())
    assert image_vec.shape[-1] == 2
    assert torch.allclose(
        torch.linalg.vector_norm(image_vec, dim=-1), torch.ones(1), atol=1e-5
    )

    text_vec = service.encode_texts(("cat",))
    assert [call[0] for call in load_calls] == ["processor", "model"]
    assert text_vec.shape[-1] == 2

    empty = service.encode_texts(())
    assert empty.numel() == 0


def test_siglip_internal_helpers_and_empty_image_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = object.__new__(siglip_mod.SiglipService)
    service.device = torch.device("cpu")

    assert siglip_mod.SiglipService._load_flag("yes") is True
    assert siglip_mod.SiglipService._load_flag("off") is False

    feature = SimpleNamespace(image_embeds=None, text_embeds=torch.tensor([2.0]))
    assert torch.allclose(
        siglip_mod.SiglipService._as_feature_tensor(feature),
        torch.tensor([2.0]),
    )
    assert siglip_mod.SiglipService._as_feature_tensor(
        (torch.tensor([3.0]),)
    ).tolist() == [3.0]
    image = SimpleNamespace()
    assert siglip_mod.SiglipService._normalize_image(image) is image

    monkeypatch.setattr(
        siglip_mod.torch, "empty", lambda *args, **kwargs: torch.tensor([])
    )
    assert service.encode_images(()).numel() == 0


def test_siglip_device_resolution_and_image_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(siglip_mod.torch.cuda, "is_available", lambda: True)
    assert str(siglip_mod._resolve_device("cuda")) == "cuda"

    service = object.__new__(siglip_mod.SiglipService)
    service.device = torch.device("cpu")
    assert siglip_mod.SiglipService.preload(service) is None
    assert siglip_mod.SiglipService._as_feature_tensor(object()) is not None

    class _Image:
        def convert(self, mode: str):
            return mode

    assert siglip_mod.SiglipService._normalize_image(_Image()) == "RGB"
