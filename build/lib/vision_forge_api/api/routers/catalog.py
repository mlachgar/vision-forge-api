"""Tag catalog endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ..context import AppContext


router = APIRouter(tags=["catalog"])


class TagSetDetail(BaseModel):
    name: str
    description: str | None = None
    canonical_tags: list[str]


class TagSetsList(BaseModel):
    tag_sets: list[TagSetDetail]
    total: int


class ProfileDetail(BaseModel):
    name: str
    description: str | None = None
    tag_sets: list[str]
    canonical_tags: list[str]


class ProfilesList(BaseModel):
    profiles: list[ProfileDetail]
    total: int


@router.get("/tag-sets", response_model=TagSetsList)
def list_tag_sets(request: Request) -> TagSetsList:
    context: AppContext = request.app.state.context
    catalog = context.tag_catalog
    items = [
        TagSetDetail(
            name=tag_set.name,
            description=tag_set.description,
            canonical_tags=list(tag_set.canonical_tags),
        )
        for tag_set in catalog.list_tag_sets()
    ]
    return TagSetsList(tag_sets=items, total=len(items))


@router.get("/profiles", response_model=ProfilesList)
def list_profiles(request: Request) -> ProfilesList:
    context: AppContext = request.app.state.context
    catalog = context.tag_catalog
    items: list[ProfileDetail] = []
    for profile in catalog.list_profiles():
        detail = catalog.profile_detail(profile.name)
        items.append(
            ProfileDetail(
                name=profile.name,
                description=profile.description,
                tag_sets=list(profile.tag_sets),
                canonical_tags=list(detail.canonical_tags),
            )
        )
    return ProfilesList(profiles=items, total=len(items))
