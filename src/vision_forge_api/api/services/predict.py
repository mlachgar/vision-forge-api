"""Request orchestration for prediction endpoints."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence

from PIL import Image
from fastapi import UploadFile

from ..context import AppContext
from ..errors import BadRequestError


@dataclass(frozen=True)
class PreparedPredictionOptions:
    canonical_tags: tuple[str, ...]
    extra_tags: tuple[str, ...]
    selected_tag_sets: tuple[str, ...]
    resolved_profile: str
    limit: int
    min_score: float


@dataclass(frozen=True)
class PreparedPredictionRequest:
    image: Image.Image
    canonical_tags: tuple[str, ...]
    extra_tags: tuple[str, ...]
    selected_tag_sets: tuple[str, ...]
    resolved_profile: str
    limit: int
    min_score: float


class PredictRequestService:
    """Validates and normalizes request inputs for prediction scoring."""

    DEFAULT_PROFILE = "default"

    def __init__(self, context: AppContext) -> None:
        self._context = context

    def build_options(
        self,
        limit: int | None,
        min_score: float | None,
        profile: str | None,
        tag_sets: str | None,
        extra_tags: str | None,
    ) -> PreparedPredictionOptions:
        tag_set_names = self._split_csv(tag_sets)
        resolved_profile = profile or self.DEFAULT_PROFILE
        canonical_tags = self._resolve_canonical_tags(
            profile=resolved_profile, tag_sets=tag_sets
        )
        extras = tuple(self._split_csv(extra_tags))

        limit_value = self._resolve_limit(limit)
        min_score_value = self._resolve_min_score(
            min_score, self._context.settings.default_min_score
        )

        return PreparedPredictionOptions(
            canonical_tags=canonical_tags,
            extra_tags=extras,
            selected_tag_sets=tuple(tag_set_names),
            resolved_profile=resolved_profile,
            limit=limit_value,
            min_score=min_score_value,
        )

    def build_request(
        self,
        file: UploadFile,
        limit: int | None,
        min_score: float | None,
        profile: str | None,
        tag_sets: str | None,
        extra_tags: str | None,
    ) -> PreparedPredictionRequest:
        options = self.build_options(
            limit=limit,
            min_score=min_score,
            profile=profile,
            tag_sets=tag_sets,
            extra_tags=extra_tags,
        )
        image = self._decode_image(file)

        return PreparedPredictionRequest(
            image=image,
            canonical_tags=options.canonical_tags,
            extra_tags=options.extra_tags,
            selected_tag_sets=options.selected_tag_sets,
            resolved_profile=options.resolved_profile,
            limit=options.limit,
            min_score=options.min_score,
        )

    def _resolve_canonical_tags(
        self, profile: str | None, tag_sets: str | None
    ) -> tuple[str, ...]:
        canonical_tags: list[str] = []
        if profile:
            try:
                detail = self._context.tag_catalog.profile_detail(profile)
            except KeyError as exc:
                raise BadRequestError(str(exc)) from exc
            canonical_tags.extend(detail.canonical_tags)

        tag_set_names = self._split_csv(tag_sets)
        if tag_set_names:
            canonical_tags.extend(self._collect_tag_set_tags(tag_set_names))

        return tuple(canonical_tags)

    def _collect_tag_set_tags(self, names: Sequence[str]) -> list[str]:
        resolved: OrderedDict[str, None] = OrderedDict()
        for name in names:
            try:
                canonical = self._context.tag_catalog.resolve_canonical_tags((name,))
            except KeyError as exc:
                raise BadRequestError(str(exc)) from exc
            for tag in canonical:
                resolved.setdefault(tag, None)
        return list(resolved)

    @staticmethod
    def _split_csv(value: str | None) -> list[str]:
        if not value:
            return []
        return [item.strip() for item in value.split(",") if item.strip()]

    def _resolve_limit(self, limit: int | None) -> int:
        if limit is not None and limit <= 0:
            raise BadRequestError("limit must be a positive integer")
        candidate = limit if limit is not None else self._context.settings.default_limit
        return min(max(1, candidate), self._context.settings.max_limit)

    @staticmethod
    def _decode_image(file: UploadFile) -> Image.Image:
        file.file.seek(0)
        try:
            return Image.open(file.file).convert("RGB")
        except Exception as exc:
            raise BadRequestError("Unable to decode image") from exc

    @staticmethod
    def _resolve_min_score(min_score: float | None, default: float = 0.0) -> float:
        value = default if min_score is None else min_score
        if value < 0.0 or value > 1.0:
            raise BadRequestError("min_score must be between 0.0 and 1.0")
        return value
