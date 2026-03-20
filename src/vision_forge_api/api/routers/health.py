"""Health check and diagnostics endpoints."""

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ..context import AppContext

router = APIRouter()


class HealthMeta(BaseModel):
    app_name: str
    version: str


class HealthResponse(BaseModel):
    status: str
    meta: HealthMeta


@router.get("/health", tags=["health"])
def health(request: Request) -> HealthResponse:
    context: AppContext = request.app.state.context
    return HealthResponse(
        status="ok",
        meta=HealthMeta(app_name=context.settings.app_name, version=context.version),
    )
