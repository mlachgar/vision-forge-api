"""Prediction endpoint wiring."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Request, UploadFile
from pydantic import BaseModel, Field

from ...auth.deps import require_predict
from ...auth.models import ApiKeyEntry
from ..context import AppContext
from ..services.predict import PredictRequestService


router = APIRouter(tags=["predict"])


class TagResult(BaseModel):
    label: str
    score: float = Field(ge=0.0, le=1.0)


class PredictMeta(BaseModel):
    profile: str
    tag_sets: list[str]
    extra_tags: list[str]
    limit: int
    min_score: float


class PredictResponse(BaseModel):
    tags: list[TagResult]
    meta: PredictMeta
    caption: str | None = None


@router.post("/predict")
async def predict(
    request: Request,
    _: Annotated[ApiKeyEntry, Depends(require_predict)],
    file: Annotated[UploadFile, File(...)],
    limit: int | None = None,
    min_score: float | None = None,
    profile: str | None = None,
    tag_sets: str | None = None,
    extra_tags: str | None = None,
    include_caption: bool = False,
) -> PredictResponse:
    context: AppContext = request.app.state.context
    request_service = PredictRequestService(context)
    prepared = request_service.build_request(
        file=file,
        limit=limit,
        min_score=min_score,
        profile=profile,
        tag_sets=tag_sets,
        extra_tags=extra_tags,
        include_caption=include_caption,
    )

    predictions = context.prediction_service.score_image(
        image=prepared.image,
        canonical_tags=prepared.canonical_tags,
        extra_labels=prepared.extra_tags,
        min_score=prepared.min_score,
        limit=prepared.limit,
    )
    caption = (
        context.prediction_service.build_caption(predictions)
        if prepared.include_caption
        else None
    )

    return PredictResponse(
        tags=[
            TagResult(label=prediction.canonical_tag, score=prediction.score)
            for prediction in predictions
        ],
        meta=PredictMeta(
            profile=prepared.resolved_profile,
            tag_sets=list(prepared.selected_tag_sets),
            extra_tags=list(prepared.extra_tags),
            limit=prepared.limit,
            min_score=prepared.min_score,
        ),
        caption=caption,
    )
