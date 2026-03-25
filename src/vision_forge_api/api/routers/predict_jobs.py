"""Asynchronous prediction job endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Request, UploadFile, status
from pydantic import BaseModel

from ...auth.deps import require_predict
from ...auth.models import ApiKeyEntry
from ..context import AppContext
from ..services.predict import PredictRequestService
from ..services.predict_jobs import PredictJobRecord


router = APIRouter(prefix="/predict", tags=["predict"])


class PredictJobTagResult(BaseModel):
    label: str
    score: float


class PredictJobItemResponse(BaseModel):
    item_id: str
    filename: str
    status: str
    tags: list[PredictJobTagResult]
    caption: str | None = None
    error: str | None = None


class PredictJobResponse(BaseModel):
    job_id: str
    status: str
    total_items: int
    completed_items: int
    failed_items: int
    items: list[PredictJobItemResponse]


def _service(request: Request):
    context: AppContext = request.app.state.context
    if context.prediction_job_service is None:
        raise RuntimeError("prediction job service is unavailable")
    return context.prediction_job_service


def _to_response(record: PredictJobRecord) -> PredictJobResponse:
    return PredictJobResponse(
        job_id=record.job_id,
        status=record.status,
        total_items=record.total_items,
        completed_items=record.completed_items,
        failed_items=record.failed_items,
        items=[
            PredictJobItemResponse(
                item_id=item.item_id,
                filename=item.filename,
                status=item.status,
                tags=[
                    PredictJobTagResult(label=tag, score=score)
                    for tag, score in item.tags
                ],
                caption=item.caption,
                error=item.error,
            )
            for item in record.items
        ],
    )


@router.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
async def submit_predict_job(
    request: Request,
    _: Annotated[ApiKeyEntry, Depends(require_predict)],
    files: Annotated[list[UploadFile], File(...)],
    limit: int | None = None,
    min_score: float | None = None,
    profile: str | None = None,
    tag_sets: str | None = None,
    extra_tags: str | None = None,
    include_caption: bool = False,
) -> PredictJobResponse:
    context: AppContext = request.app.state.context
    request_service = PredictRequestService(context)
    options = request_service.build_options(
        limit=limit,
        min_score=min_score,
        profile=profile,
        tag_sets=tag_sets,
        extra_tags=extra_tags,
        include_caption=include_caption,
    )
    service = _service(request)
    record = await service.submit_job(
        files=files,
        options=options,
    )
    return _to_response(record)


@router.get("/jobs/{job_id}")
def get_predict_job(
    job_id: str,
    request: Request,
    _: Annotated[ApiKeyEntry, Depends(require_predict)],
) -> PredictJobResponse:
    return _to_response(_service(request).get_job(job_id))


@router.delete("/jobs/{job_id}")
def cancel_predict_job(
    job_id: str,
    request: Request,
    _: Annotated[ApiKeyEntry, Depends(require_predict)],
) -> PredictJobResponse:
    return _to_response(_service(request).cancel_job(job_id))
