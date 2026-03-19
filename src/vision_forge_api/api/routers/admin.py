"""Admin API key management and reload endpoints."""

from typing import Iterable, Sequence

from fastapi import APIRouter, Depends, Request, Response, status
from pydantic import BaseModel

from ...auth.deps import require_admin
from ...auth.models import ApiKeyEntry
from ...config.schema import AuthRole
from ..context import AppContext
from ..services.admin import AdminService


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
    return [
        ApiKeySummary(name=entry.name, roles=list(entry.roles), enabled=entry.enabled)
        for entry in entries
    ]


def _service_from_request(request: Request) -> AdminService:
    context: AppContext = request.app.state.context
    return AdminService(context)


@router.get("/api-keys", response_model=list[ApiKeySummary])
def list_api_keys(
    request: Request, _: ApiKeyEntry = Depends(require_admin)
) -> list[ApiKeySummary]:
    return _summaries(_service_from_request(request).list_api_keys())


@router.post(
    "/api-keys",
    response_model=ApiKeyCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_api_key(
    payload: ApiKeyCreateRequest,
    request: Request,
    _: ApiKeyEntry = Depends(require_admin),
) -> ApiKeyCreateResponse:
    created = _service_from_request(request).create_api_key(
        name=payload.name,
        roles=tuple(payload.roles) if payload.roles is not None else None,
        enabled=payload.enabled,
    )
    return ApiKeyCreateResponse(
        name=created.name,
        token=created.token,
        roles=list(created.roles),
        enabled=created.enabled,
    )


@router.patch("/api-keys/{name}", response_model=ApiKeySummary)
def update_api_key(
    name: str,
    payload: ApiKeyUpdateRequest,
    request: Request,
    _: ApiKeyEntry = Depends(require_admin),
) -> ApiKeySummary:
    updated = _service_from_request(request).update_api_key_enabled(
        name=name, enabled=payload.enabled
    )
    return ApiKeySummary(
        name=updated.name, roles=list(updated.roles), enabled=updated.enabled
    )


@router.delete(
    "/api-keys/{name}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response
)
def delete_api_key(
    name: str, request: Request, _: ApiKeyEntry = Depends(require_admin)
) -> Response:
    _service_from_request(request).delete_api_key(name)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/reload", response_model=ReloadResponse)
def reload_configuration(
    request: Request, _: ApiKeyEntry = Depends(require_admin)
) -> ReloadResponse:
    service = _service_from_request(request)
    new_context = service.reload_configuration()
    request.app.state.context = new_context
    return ReloadResponse(status="ok", version=new_context.version)
