from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from starlette.requests import Request

import vision_forge_api.api.routers.predict_jobs as router_mod
from vision_forge_api.api.services.predict import PreparedPredictionOptions
from vision_forge_api.api.services.predict_jobs import (
    PredictJobItemResult,
    PredictJobRecord,
)
from vision_forge_api.auth.models import ApiKeyEntry
from vision_forge_api.config.schema import AuthRole


def _request(app: FastAPI) -> Request:
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": [],
        "app": app,
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    return Request(scope)


def _api_entry() -> ApiKeyEntry:
    return ApiKeyEntry(
        name="tester",
        key_hash="sha256:abc",
        roles=(AuthRole.PREDICT,),
        enabled=True,
    )


def _job_record() -> PredictJobRecord:
    return PredictJobRecord(
        job_id="job-1",
        status="done",
        total_items=1,
        completed_items=1,
        failed_items=0,
        items=[
            PredictJobItemResult(
                item_id="item-1",
                filename="one.jpg",
                status="done",
                tags=[("cat", 0.91)],
                caption="caption",
            )
        ],
    )


def test_to_response_maps_nested_items() -> None:
    response = router_mod._to_response(_job_record())

    assert response.job_id == "job-1"
    assert response.items[0].tags[0].label == "cat"
    assert response.items[0].caption == "caption"


@pytest.mark.asyncio
async def test_predict_job_router_endpoints_and_missing_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FastAPI()
    app.state.context = SimpleNamespace(
        prediction_job_service=None,
        settings=SimpleNamespace(default_limit=5, max_limit=10, default_min_score=0.1),
        tag_catalog=SimpleNamespace(),
    )
    request = _request(app)

    with pytest.raises(RuntimeError):
        router_mod._service(request)

    captured: dict[str, object] = {}

    class _Service:
        async def submit_job(self, *, files, options):
            captured["files"] = files
            captured["options"] = options
            return _job_record()

        def get_job(self, job_id):
            captured["get_job"] = job_id
            return _job_record()

        def cancel_job(self, job_id):
            captured["cancel_job"] = job_id
            return _job_record()

    class _RequestService:
        def __init__(self, context) -> None:
            captured["context"] = context

        def build_options(self, **kwargs):
            captured["build_options"] = kwargs
            return PreparedPredictionOptions(
                canonical_tags=("cat",),
                extra_tags=("bonus",),
                selected_tag_sets=("animals",),
                resolved_profile="default",
                limit=2,
                min_score=0.1,
                include_caption=True,
            )

    app.state.context.prediction_job_service = _Service()
    monkeypatch.setattr(router_mod, "PredictRequestService", _RequestService)

    uploaded = SimpleNamespace(filename="one.jpg")
    response = await router_mod.submit_predict_job(
        request=request,
        _=_api_entry(),
        files=[uploaded],
        limit=2,
        min_score=0.1,
        profile="default",
        tag_sets="animals",
        extra_tags="bonus",
        include_caption=True,
    )
    assert response.job_id == "job-1"
    assert captured["build_options"]["include_caption"] is True

    got = router_mod.get_predict_job("job-1", request, _api_entry())
    assert got.items[0].tags[0].label == "cat"
    assert captured["get_job"] == "job-1"

    canceled = router_mod.cancel_predict_job("job-1", request, _api_entry())
    assert canceled.status == "done"
    assert captured["cancel_job"] == "job-1"
