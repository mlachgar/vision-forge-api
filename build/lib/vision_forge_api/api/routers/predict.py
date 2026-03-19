"""Prediction endpoint wiring."""

from fastapi import APIRouter, Depends, File, Request, UploadFile
from pydantic import BaseModel, Field

from vision_forge_api.auth.deps import require_predict
from vision_forge_api.auth.models import ApiKeyEntry
from ..context import AppContext
from ..services.predict import PredictRequestService


router = APIRouter(tags=["predict"])


class TagResult(BaseModel):
    label: str
    score: float = Field(ge=-1.0, le=1.0)


class PredictMeta(BaseModel):
    profile: str | None = None
    tag_sets: list[str]
    extra_tags: list[str]
    limit: int
    min_score: float


class PredictResponse(BaseModel):
    tags: list[TagResult]
    meta: PredictMeta


@router.post("/predict", response_model=PredictResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    limit: int | None = None,
    min_score: float | None = None,
    profile: str | None = None,
    tag_sets: str | None = None,
    extra_tags: str | None = None,
    _: ApiKeyEntry = Depends(require_predict),
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
    )

    predictions = context.prediction_service.score_image(
        image=prepared.image,
        canonical_tags=prepared.canonical_tags,
        extra_labels=prepared.extra_tags,
        min_score=prepared.min_score,
        limit=prepared.limit,
    )

    return PredictResponse(
        tags=[
            TagResult(label=prediction.canonical_tag, score=prediction.score)
            for prediction in predictions
        ],
        meta=PredictMeta(
            profile=profile,
            tag_sets=list(prepared.selected_tag_sets),
            extra_tags=list(prepared.extra_tags),
            limit=prepared.limit,
            min_score=prepared.min_score,
        ),
    )
