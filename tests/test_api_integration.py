from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vision_forge_api.api.errors import register_exception_handlers
from vision_forge_api.api.routers.admin import router as admin_router
from vision_forge_api.api.routers.catalog import router as catalog_router
from vision_forge_api.api.routers.health import router as health_router
from vision_forge_api.api.routers.predict import router as predict_router
from vision_forge_api.api.services.admin import AdminService
from vision_forge_api.auth.cache import ApiKeyRepository, AuthCache, hash_token
from vision_forge_api.auth.models import ApiKeyEntry
from vision_forge_api.catalog.service import TagCatalog
from vision_forge_api.config.loader import ConfigLoader
from vision_forge_api.config.schema import AuthRole


ROOT = Path(__file__).resolve().parents[1]
ADMIN_TOKEN = "vfk_admin_token_12345678abcdef12"
PREDICT_TOKEN = "vfk_predict_token_87654321fedcba"
CREATED_TOKEN = "vfk_" + "a" * 28


class _PredictionServiceStub:
    def __init__(self, catalog: TagCatalog) -> None:
        self._catalog = catalog

    def score_image(
        self,
        image,
        canonical_tags,
        extra_labels,
        min_score,
        limit,
    ):
        labels = list(dict.fromkeys([*canonical_tags, *extra_labels]))
        if not labels:
            labels = list(self._catalog.canonical_tags()[:1]) or ["fallback"]
        rotations = (image.width + image.height) % len(labels) if labels else 0
        ordered = labels[rotations:] + labels[:rotations]
        top_labels = ordered[: max(1, min(limit, len(ordered)))]
        return [
            SimpleNamespace(
                canonical_tag=label,
                score=round(0.99 - index * 0.01, 4),
                is_extra=label in extra_labels,
            )
            for index, label in enumerate(top_labels)
            if 0.99 - index * 0.01 >= min_score
        ]


def _build_context(tmp_path: Path) -> SimpleNamespace:
    loader = ConfigLoader(ROOT / "config")
    settings = loader.load_settings()
    auth_config = loader.load_auth()
    tag_catalog = TagCatalog(
        loader.load_tag_sets(), loader.load_profiles(), loader.load_prompts()
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    temp_repo = ApiKeyRepository(data_dir)
    temp_repo.persist(ApiKeyRepository(ROOT / "data").read_all())

    auth_cache = AuthCache.from_repository(temp_repo)

    class _TokenManagerStub:
        def generate_token(self) -> str:
            return CREATED_TOKEN

        def hash_token(self, token: str) -> str:
            return hash_token(token)

    return SimpleNamespace(
        loader=loader,
        settings=settings.model_copy(
            update={
                "embeddings_dir": data_dir / "embeddings",
                "model_cache_dir": data_dir / "model_cache",
            }
        ),
        auth_config=auth_config,
        version="test",
        config_dir=ROOT / "config",
        tag_catalog=tag_catalog,
        auth_cache=auth_cache,
        token_manager=_TokenManagerStub(),
        api_key_repo=temp_repo,
        siglip_service=SimpleNamespace(),
        prediction_service=_PredictionServiceStub(tag_catalog),
    )


def _build_app(context: SimpleNamespace) -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)
    app.state.context = context
    app.include_router(health_router)
    app.include_router(catalog_router)
    app.include_router(predict_router)
    app.include_router(admin_router)
    return app


@pytest.fixture()
def app(tmp_path: Path) -> FastAPI:
    return _build_app(_build_context(tmp_path))


@pytest.fixture()
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


@pytest.fixture()
def admin_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {ADMIN_TOKEN}"}


@pytest.fixture()
def predict_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {PREDICT_TOKEN}"}


@pytest.fixture()
def admin_only_headers(app: FastAPI) -> dict[str, str]:
    admin_only = ApiKeyEntry(
        name="admin-only",
        key_hash=hash_token("admin-only-token"),
        roles=(AuthRole.ADMIN,),
        enabled=True,
    )
    app.state.context.auth_cache.reload(
        (*app.state.context.auth_cache.entries, admin_only)
    )
    return {"Authorization": "Bearer admin-only-token"}


def test_public_endpoints_return_expected_catalog_data(
    client: TestClient, app: FastAPI
) -> None:
    loader = ConfigLoader(ROOT / "config")
    expected_tag_sets = loader.load_tag_sets().tag_sets
    expected_profiles = loader.load_profiles().profiles

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "meta": {
            "app_name": app.state.context.settings.app_name,
            "version": app.state.context.version,
        },
    }

    tag_sets = client.get("/tag-sets")
    assert tag_sets.status_code == 200
    tag_sets_payload = tag_sets.json()
    assert tag_sets_payload["total"] == len(expected_tag_sets)
    assert tag_sets_payload["tag_sets"][0]["name"] == expected_tag_sets[0].name

    profiles = client.get("/profiles")
    assert profiles.status_code == 200
    profiles_payload = profiles.json()
    assert profiles_payload["total"] == len(expected_profiles)
    assert profiles_payload["profiles"][0]["name"] == expected_profiles[0].name


@pytest.mark.parametrize(
    "sample_name",
    [
        "cat.jpg",
        "coffee.jpg",
        "vehicle_car.jpg",
        "white_house_night.jpg",
    ],
)
def test_predict_accepts_sample_images(
    client: TestClient,
    app: FastAPI,
    predict_headers: dict[str, str],
    sample_name: str,
) -> None:
    sample_path = ROOT / "samples" / sample_name
    assert sample_path.is_file(), sample_path
    profile_name = app.state.context.tag_catalog.list_profiles()[0].name
    tag_set_name = app.state.context.tag_catalog.list_tag_sets()[0].name

    response = client.post(
        "/predict",
        headers=predict_headers,
        params={"profile": profile_name, "tag_sets": tag_set_name, "limit": 3},
        files={"file": (sample_path.name, sample_path.read_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["meta"]["profile"] == profile_name
    assert payload["meta"]["limit"] == 3
    assert payload["meta"]["tag_sets"] == [tag_set_name]
    assert 1 <= len(payload["tags"]) <= 3
    assert payload["tags"] == sorted(
        payload["tags"], key=lambda item: item["score"], reverse=True
    )
    for item in payload["tags"]:
        assert -1.0 <= item["score"] <= 1.0


def test_predict_requires_predict_role(
    client: TestClient, app: FastAPI, admin_only_headers: dict[str, str]
) -> None:
    sample_path = ROOT / "samples" / "cat.jpg"
    profile_name = app.state.context.tag_catalog.list_profiles()[0].name
    response = client.post(
        "/predict",
        headers=admin_only_headers,
        params={"profile": profile_name, "limit": 2},
        files={"file": (sample_path.name, sample_path.read_bytes(), "image/jpeg")},
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "API key lacks the required role"}


def test_admin_crud_and_reload(
    client: TestClient,
    app: FastAPI,
    admin_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = client.get("/admin/api-keys", headers=admin_headers)
    assert initial.status_code == 200
    initial_names = {item["name"] for item in initial.json()}
    assert {"admin", "predictor"}.issubset(initial_names)

    created = client.post(
        "/admin/api-keys",
        headers=admin_headers,
        json={"name": "integration-created"},
    )
    assert created.status_code == 201
    created_payload = created.json()
    assert created_payload == {
        "name": "integration-created",
        "token": CREATED_TOKEN,
        "roles": ["predict"],
        "enabled": True,
    }

    after_create = client.get("/admin/api-keys", headers=admin_headers)
    assert after_create.status_code == 200
    after_create_names = {item["name"] for item in after_create.json()}
    assert "integration-created" in after_create_names

    updated = client.patch(
        "/admin/api-keys/integration-created",
        headers=admin_headers,
        json={"enabled": False},
    )
    assert updated.status_code == 200
    assert updated.json()["enabled"] is False

    deleted = client.delete(
        "/admin/api-keys/integration-created", headers=admin_headers
    )
    assert deleted.status_code == 204

    after_delete = client.get("/admin/api-keys", headers=admin_headers)
    assert after_delete.status_code == 200
    assert "integration-created" not in {item["name"] for item in after_delete.json()}

    new_context = SimpleNamespace(**{**vars(app.state.context), "version": "2.0.0"})
    monkeypatch.setattr(AdminService, "reload_configuration", lambda self: new_context)

    reloaded = client.post("/admin/reload", headers=admin_headers)
    assert reloaded.status_code == 200
    assert reloaded.json() == {"status": "ok", "version": "2.0.0"}
    assert app.state.context is new_context
