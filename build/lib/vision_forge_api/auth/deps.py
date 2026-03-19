"""Authentication dependencies for FastAPI routers."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status

from ..auth.cache import AuthError, parse_authorization_header
from ..auth.models import ApiKeyEntry
from ..config.schema import AuthRole
from ..api.context import AppContext


def _context_from_request(request: Request) -> AppContext:
    return request.app.state.context


def require_api_key(
    request: Request,
    required_role: AuthRole | None = None,
    context: AppContext | None = Depends(_context_from_request),
) -> ApiKeyEntry:
    header_value = request.headers.get("authorization")
    token = parse_authorization_header(header_value)
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required",
        )
    try:
        return context.auth_cache.authorize(token, required_role=required_role)
    except AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc


def require_admin(request: Request, context: AppContext = Depends(_context_from_request)) -> ApiKeyEntry:
    return require_api_key(request, required_role=AuthRole.ADMIN, context=context)
