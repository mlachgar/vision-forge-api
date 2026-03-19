"""Admin API key management and reload endpoints."""

from __future__ import annotations

from typing import Iterable, Sequence

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel

from vision_forge_api.auth.deps import require_admin
from vision_forge_api.auth.models import ApiKeyEntry
from vision_forge_api.config.schema import AuthRole
from ..context import AppContext
from ..context_builder import build_context


router = APIRouter(prefix="/admin", tags=["admin"])


class ApiKeySummary(BaseModel):
    name: str
    roles: list[AuthRole]
    enabled: bool


class ApiKeyCreateRequest(BaseModel):
    name: str
    roles: Sequence[AuthRole] | None = None
    enabled: bool | None = None


class ApiKeyCreateResponse(BaseModel):
    name: str
    token: str
    roles: list[AuthRole]
    enabled: bool


class ApiKeyUpdateRequest(BaseModel):
    enabled: bool | None = None


class ReloadResponse(BaseModel):
    status: str
    version: str


def _summaries(entries: Iterable[ApiKeyEntry]) -> list[ApiKeySummary]:
    return [ApiKeySummary(name=entry.name, roles=list(entry.roles), enabled=entry.enabled) for entry in entries]


@router.get("/api-keys", response_model=list[ApiKeySummary])
def list_api_keys(request: Request, _: ApiKeyEntry = Depends(require_admin)) -> list[ApiKeySummary]:
    context: AppContext = request.app.state.context
    return _summaries(context.auth_cache.entries)


@router.post("/api-keys", response_model=ApiKeyCreateResponse, status_code=status.HTTP_201_CREATED)
def create_api_key(payload: ApiKeyCreateRequest, request: Request, _: ApiKeyEntry = Depends(require_admin)) -> ApiKeyCreateResponse:
    context: AppContext = request.app.state.context
    entries = context.api_key_repo.read_all()
    if any(entry.name == payload.name for entry in entries):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="API key with that name already exists")
    roles = payload.roles if payload.roles is not None else context.auth_config.default_roles
    token = context.token_manager.generate_token()
    key_hash = context.token_manager.hash_token(token)
    enabled = payload.enabled if payload.enabled is not None else True
    new_entry = ApiKeyEntry(name=payload.name, key_hash=key_hash, roles=roles, enabled=enabled)
    entries.append(new_entry)
    context.api_key_repo.persist(entries)
    context.auth_cache.reload(entries)
    return ApiKeyCreateResponse(name=new_entry.name, token=token, roles=list(new_entry.roles), enabled=new_entry.enabled)


@router.patch("/api-keys/{name}", response_model=ApiKeySummary)
def update_api_key(name: str, payload: ApiKeyUpdateRequest, request: Request, _: ApiKeyEntry = Depends(require_admin)) -> ApiKeySummary:
    context: AppContext = request.app.state.context
    entries = context.api_key_repo.read_all()
    for idx, entry in enumerate(entries):
        if entry.name != name:
            continue
        updated = entry
        if payload.enabled is not None:
            updated = entry.model_copy(update={"enabled": payload.enabled})
        entries[idx] = updated
        context.api_key_repo.persist(entries)
        context.auth_cache.reload(entries)
        return ApiKeySummary(name=updated.name, roles=list(updated.roles), enabled=updated.enabled)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found")


@router.delete("/api-keys/{name}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_api_key(name: str, request: Request, _: ApiKeyEntry = Depends(require_admin)) -> Response:
    context: AppContext = request.app.state.context
    entries = context.api_key_repo.read_all()
    remaining = [entry for entry in entries if entry.name != name]
    if len(remaining) == len(entries):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found")
    context.api_key_repo.persist(remaining)
    context.auth_cache.reload(remaining)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@router.post("/reload", response_model=ReloadResponse)
def reload_configuration(request: Request, _: ApiKeyEntry = Depends(require_admin)) -> ReloadResponse:
    context: AppContext = request.app.state.context
    new_context = build_context(context.loader, context.version)
    request.app.state.context = new_context
    return ReloadResponse(status="ok", version=new_context.version)
