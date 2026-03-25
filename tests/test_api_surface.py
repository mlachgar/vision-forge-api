from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, UploadFile
from PIL import Image
from starlette.requests import Request

from vision_forge_api.api import app as app_mod
from vision_forge_api.api.errors import BadRequestError
from vision_forge_api.api.routers import admin as admin_router_mod
from vision_forge_api.api.routers import catalog as catalog_router_mod
from vision_forge_api.api.routers import health as health_router_mod
from vision_forge_api.api.routers import predict as predict_router_mod
from vision_forge_api.api.services.admin import AdminService
from vision_forge_api.api.services.predict import PredictRequestService
from vision_forge_api.auth import deps as auth_deps
from vision_forge_api.auth.cache import AuthorizationResult
from vision_forge_api.auth.models import ApiKeyEntry
from vision_forge_api.config.schema import AuthRole


def _request(
    app: FastAPI, *, headers: list[tuple[bytes, bytes]] | None = None
) -> Request:
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": headers or [],
        "app": app,
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    return Request(scope)


def _image_upload(filename: str = "img.png") -> UploadFile:
    buf = BytesIO()
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(buf, format="PNG")
    buf.seek(0)
    return UploadFile(filename=filename, file=buf)


def _api_entry() -> ApiKeyEntry:
    return ApiKeyEntry(
        name="tester",
        key_hash="sha256:abc",
        roles=(AuthRole.ADMIN, AuthRole.PREDICT),
        enabled=True,
    )


def test_predict_request_service_build_request_and_validation() -> None:
    tag_catalog = SimpleNamespace(
        profile_detail=lambda profile: (
            SimpleNamespace(canonical_tags=("cat", "dog"))
            if profile == "default"
            else (_ for _ in ()).throw(KeyError("bad profile"))
        ),
        resolve_canonical_tags=lambda names: tuple(
            "cat" if n == "animals" else (_ for _ in ()).throw(KeyError("bad set"))
            for n in names
        ),
    )
    context = SimpleNamespace(
        settings=SimpleNamespace(default_limit=5, max_limit=10, default_min_score=0.1),
        tag_catalog=tag_catalog,
    )
    service = PredictRequestService(context)

    prepared = service.build_request(
        file=_image_upload(),
        limit=100,
        min_score=None,
        profile="default",
        tag_sets="animals",
        extra_tags="x,y",
    )
    assert prepared.canonical_tags == ("cat", "dog", "cat")
    assert prepared.extra_tags == ("x", "y")
    assert prepared.selected_tag_sets == ("animals",)
    assert prepared.resolved_profile == "default"
    assert prepared.limit == 10
    assert prepared.min_score == 0.1

    default_prepared = service.build_request(
        file=_image_upload(),
        limit=None,
        min_score=None,
        profile=None,
        tag_sets=None,
        extra_tags=None,
    )
    assert default_prepared.canonical_tags == ("cat", "dog")
    assert default_prepared.resolved_profile == "default"

    with pytest.raises(BadRequestError):
        service.build_request(
            file=_image_upload(),
            limit=0,
            min_score=0.0,
            profile="default",
            tag_sets=None,
            extra_tags=None,
        )

    with pytest.raises(BadRequestError):
        service.build_request(
            file=_image_upload(),
            limit=1,
            min_score=1.2,
            profile="default",
            tag_sets=None,
            extra_tags=None,
        )

    with pytest.raises(BadRequestError):
        service.build_request(
            file=_image_upload(),
            limit=1,
            min_score=-0.1,
            profile="default",
            tag_sets=None,
            extra_tags=None,
        )

    bad_upload = UploadFile(filename="bad.txt", file=BytesIO(b"not-an-image"))
    with pytest.raises(BadRequestError):
        service.build_request(
            file=bad_upload,
            limit=1,
            min_score=0.0,
            profile="default",
            tag_sets=None,
            extra_tags=None,
        )


def test_predict_request_service_collects_tag_set_errors() -> None:
    context = SimpleNamespace(
        settings=SimpleNamespace(default_limit=5, max_limit=10, default_min_score=0.1),
        tag_catalog=SimpleNamespace(
            profile_detail=lambda _profile: (_ for _ in ()).throw(
                KeyError("missing profile")
            ),
            resolve_canonical_tags=lambda _names: (_ for _ in ()).throw(
                KeyError("missing set")
            ),
        ),
    )
    service = PredictRequestService(context)

    with pytest.raises(BadRequestError):
        service._resolve_canonical_tags("missing", None)
    with pytest.raises(BadRequestError):
        service._collect_tag_set_tags(("missing",))


def test_health_and_catalog_router_functions() -> None:
    app = FastAPI()
    app.state.context = SimpleNamespace(
        version="1.2.3",
        settings=SimpleNamespace(app_name="vf-app"),
        tag_catalog=SimpleNamespace(
            list_tag_sets=lambda: (
                SimpleNamespace(
                    name="animals", description="desc", canonical_tags=("cat", "dog")
                ),
            ),
            list_profiles=lambda: (
                SimpleNamespace(name="default", description="p", tag_sets=("animals",)),
            ),
            profile_detail=lambda _name: SimpleNamespace(canonical_tags=("cat", "dog")),
        ),
    )
    request = _request(app)

    health = health_router_mod.health(request)
    assert health.status == "ok"
    assert health.meta.version == "1.2.3"

    tag_sets = catalog_router_mod.list_tag_sets(request)
    assert tag_sets.total == 1
    assert tag_sets.tag_sets[0].canonical_tags == ["cat", "dog"]

    profiles = catalog_router_mod.list_profiles(request)
    assert profiles.total == 1
    assert profiles.profiles[0].tag_sets == ["animals"]


def test_admin_router_functions_and_service_from_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Repo:
        def __init__(self) -> None:
            self.entries = [
                ApiKeyEntry(
                    name="a",
                    key_hash="sha256:a",
                    roles=(AuthRole.PREDICT,),
                    enabled=True,
                )
            ]

        def read_all(self):
            return list(self.entries)

        def persist(self, entries):
            self.entries = list(entries)

    class _Cache:
        def __init__(self, entries):
            self._entries = tuple(entries)

        @property
        def entries(self):
            return self._entries

        def reload(self, entries):
            self._entries = tuple(entries)

    class _TokenManager:
        def generate_token(self) -> str:
            return "tok"

        def hash_token(self, token: str) -> str:
            return f"sha256:{token}"

    repo = _Repo()
    context = SimpleNamespace(
        auth_cache=_Cache(repo.read_all()),
        api_key_repo=repo,
        token_manager=_TokenManager(),
        auth_config=SimpleNamespace(default_roles=(AuthRole.PREDICT,)),
        loader=SimpleNamespace(),
        version="1.0.0",
    )
    app = FastAPI()
    app.state.context = context
    request = _request(app)
    admin_entry = _api_entry()

    service = admin_router_mod._service_from_request(request)
    assert isinstance(service, AdminService)

    listed = admin_router_mod.list_api_keys(request, admin_entry)
    assert len(listed) == 1

    created = admin_router_mod.create_api_key(
        admin_router_mod.ApiKeyCreateRequest(name="new"), request, admin_entry
    )
    assert created.token == "tok"

    updated = admin_router_mod.update_api_key(
        "new", admin_router_mod.ApiKeyUpdateRequest(enabled=False), request, admin_entry
    )
    assert updated.enabled is False

    response = admin_router_mod.delete_api_key("new", request, admin_entry)
    assert response.status_code == 204

    monkeypatch.setattr(
        admin_router_mod.AdminService,
        "reload_configuration",
        lambda _self: SimpleNamespace(version="9.9.9"),
    )
    reloaded = admin_router_mod.reload_configuration(request, admin_entry)
    assert reloaded.status == "ok"
    assert app.state.context.version == "9.9.9"


@pytest.mark.asyncio
async def test_predict_router_function() -> None:
    app = FastAPI()
    app.state.context = SimpleNamespace(
        settings=SimpleNamespace(default_limit=5, max_limit=10, default_min_score=0.0),
        tag_catalog=SimpleNamespace(
            profile_detail=lambda _name: SimpleNamespace(canonical_tags=("cat",)),
            resolve_canonical_tags=lambda _names: ("cat",),
        ),
        prediction_service=SimpleNamespace(
            score_image=lambda **_kwargs: [
                SimpleNamespace(canonical_tag="cat", score=0.77),
            ]
        ),
    )
    request = _request(app)

    response = await predict_router_mod.predict(
        request=request,
        file=_image_upload(),
        limit=5,
        min_score=0.2,
        profile=None,
        tag_sets="animals",
        extra_tags="extra",
        _=_api_entry(),
    )

    assert response.tags[0].label == "cat"
    assert response.meta.profile == "default"
    assert response.meta.tag_sets == ["animals"]
    assert response.meta.extra_tags == ["extra"]


def test_auth_dependencies_and_wrappers() -> None:
    app = FastAPI()
    expected = _api_entry()
    app.state.context = SimpleNamespace(
        auth_cache=SimpleNamespace(
            authorize=lambda token, required_role=None: AuthorizationResult(
                status_code=200,
                entry=expected if token == "good" else None,
                detail="",
            )
        )
    )

    req_ok = _request(app, headers=[(b"authorization", b"Bearer good")])
    out = auth_deps.require_api_key(
        req_ok, required_role=AuthRole.ADMIN, context=app.state.context
    )
    assert out.name == expected.name

    req_bad_header = _request(app, headers=[(b"authorization", b"bad")])
    with pytest.raises(Exception):
        auth_deps.require_api_key(req_bad_header, context=app.state.context)

    req_missing = _request(app)
    with pytest.raises(Exception):
        auth_deps.require_api_key(req_missing, context=app.state.context)

    assert (
        auth_deps.require_admin(req_ok, context=app.state.context).name == expected.name
    )
    assert (
        auth_deps.require_predict(req_ok, context=app.state.context).name
        == expected.name
    )


def test_create_app_and_version_resolution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}
    original_resolve_version = app_mod.resolve_version

    class _Loader:
        def __init__(self, config_dir):
            calls["config_dir"] = config_dir

    class _PredictionService:
        def warmup(self) -> None:
            calls["warmup"] = True

    class _JobService:
        async def start(self) -> None:
            calls["job_start"] = True

        async def stop(self) -> None:
            calls["job_stop"] = True

    fake_context = SimpleNamespace(
        settings=SimpleNamespace(app_name="vf"),
        version="x",
        prediction_service=_PredictionService(),
        prediction_job_service=_JobService(),
    )

    monkeypatch.setattr(app_mod, "ConfigLoader", _Loader)
    monkeypatch.setattr(app_mod, "build_context", lambda loader, version: fake_context)
    monkeypatch.setattr(
        app_mod,
        "register_exception_handlers",
        lambda app: calls.setdefault("handlers", app),
    )

    class _Logger:
        def info(self, *args):
            calls["log"] = args

    original_get_logger = app_mod.logging.getLogger
    monkeypatch.setattr(
        app_mod.logging,
        "getLogger",
        lambda name=None: (
            _Logger() if name == "vision_forge_api" else original_get_logger(name)
        ),
    )
    monkeypatch.setattr(app_mod, "resolve_version", lambda: "2.0.0")

    app = app_mod.create_app(tmp_path)
    assert app.title == "vf"
    assert app.version == "2.0.0"
    assert calls["config_dir"] == tmp_path

    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert "log" in calls
    assert calls["warmup"] is True
    assert calls["job_start"] is True
    assert calls["job_stop"] is True
    monkeypatch.setattr(app_mod, "resolve_version", original_resolve_version)

    monkeypatch.setattr(app_mod, "pkg_version", lambda _name: "3.0.0")
    assert app_mod.resolve_version() == "3.0.0"

    class _PkgErr(Exception):
        pass

    monkeypatch.setattr(app_mod, "PackageNotFoundError", _PkgErr)

    def _raise(_name: str) -> str:
        raise _PkgErr()

    monkeypatch.setattr(app_mod, "pkg_version", _raise)
    assert app_mod.resolve_version() == app_mod.CONFIG_VERSION
