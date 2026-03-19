from __future__ import annotations

from functools import partial
from types import SimpleNamespace

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from vision_forge_api.api.errors import ApiKeyNotFoundError, register_exception_handlers
from vision_forge_api.auth.cache import AuthCache, hash_token
from vision_forge_api.auth.deps import require_api_key
from vision_forge_api.auth.models import ApiKeyEntry
from vision_forge_api.config.schema import AuthRole


def _build_auth_app() -> FastAPI:
    app = FastAPI()
    entry = ApiKeyEntry(
        name="predict-only",
        key_hash=hash_token("predict-token"),
        roles=(AuthRole.PREDICT,),
        enabled=True,
    )
    app.state.context = SimpleNamespace(auth_cache=AuthCache([entry]))

    @app.get("/admin-only")
    def _admin_only(
        _: ApiKeyEntry = Depends(
            partial(require_api_key, required_role=AuthRole.ADMIN)
        ),
    ) -> dict[str, str]:
        return {"status": "ok"}

    return app


def test_require_api_key_missing_token_returns_401() -> None:
    client = TestClient(_build_auth_app())

    response = client.get("/admin-only")

    assert response.status_code == 401
    assert response.json()["detail"] == "Bearer token required"


def test_require_api_key_role_mismatch_returns_403() -> None:
    client = TestClient(_build_auth_app())

    response = client.get(
        "/admin-only", headers={"Authorization": "Bearer predict-token"}
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "API key lacks the required role"


def test_centralized_api_key_not_found_handler_returns_404() -> None:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/missing-key")
    def _missing_key() -> dict[str, str]:
        raise ApiKeyNotFoundError()

    response = TestClient(app).get("/missing-key")

    assert response.status_code == 404
    assert response.json() == {"detail": "API key not found"}
