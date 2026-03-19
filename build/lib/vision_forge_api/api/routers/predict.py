"""Prediction endpoint wiring."""

from __future__ import annotations

from collections import OrderedDict
from functools import partial
from typing import Sequence

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from PIL import Image

from vision_forge_api.auth.deps import require_api_key
from vision_forge_api.auth.models import ApiKeyEntry
from vision_forge_api.config.schema import AuthRole
from ..context import AppContext


router = APIRouter(tags=["predict"])


class TagResult(BaseModel):
    label: str
    score: float = Field(ge=0.0)


class PredictMeta(BaseModel):
    profile: str | None = None
    tag_sets: list[str]
    extra_tags: list[str]
    limit: int
    min_score: float


class PredictResponse(BaseModel):
    tags: list[TagResult]
    meta: PredictMeta


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _collect_tag_sets(catalog: AppContext, names: Sequence[str]) -> list[str]:
    resolved: OrderedDict[str, None] = OrderedDict()
    for name in names:
        try:
            canonical = catalog.tag_catalog.resolve_canonical_tags((name,))
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        for tag in canonical:
            resolved.setdefault(tag, None)
    return list(resolved)


@router.post("/predict", response_model=PredictResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    limit: int | None = None,
    min_score: float | None = None,
    profile: str | None = None,
    tag_sets: str | None = None,
    extra_tags: str | None = None,
    _: ApiKeyEntry = Depends(partial(require_api_key, required_role=AuthRole.PREDICT)),
) -> PredictResponse:
    context: AppContext = request.app.state.context
    selected_tag_sets: list[str] = []
    canonical_tags: list[str] = []
    if profile:
        try:
            detail = context.tag_catalog.profile_detail(profile)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        canonical_tags.extend(detail.canonical_tags)
    tag_set_names = _split_csv(tag_sets)
    if tag_set_names:
        canonical_tags.extend(_collect_tag_sets(context, tag_set_names))
        selected_tag_sets.extend(tag_set_names)
    extras = _split_csv(extra_tags)
    if not canonical_tags and not extras:
        raise HTTPException(
            status_code=400,
            detail="At least one of profile, tag_sets, or extra_tags must be provided",
        )
    if limit is not None and limit <= 0:
        raise HTTPException(status_code=400, detail="limit must be a positive integer")
    limit_value = limit or context.settings.default_limit
    limit_value = max(1, limit_value)
    if limit_value > context.settings.max_limit:
        limit_value = context.settings.max_limit
    min_score_value = min_score if min_score is not None else context.settings.default_min_score
    file.file.seek(0)
    try:
        image = Image.open(file.file).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Unable to decode image") from exc
    predictions = context.prediction_service.score_image(
        image=image,
        canonical_tags=canonical_tags,
        extra_labels=extras,
        min_score=min_score_value,
        limit=limit_value,
    )
    return PredictResponse(
        tags=[TagResult(label=prediction.canonical_tag, score=prediction.score) for prediction in predictions],
        meta=PredictMeta(
            profile=profile,
            tag_sets=selected_tag_sets,
            extra_tags=extras,
            limit=limit_value,
            min_score=min_score_value,
        ),
    )
