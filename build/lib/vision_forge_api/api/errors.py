"""Shared API error types and exception handling."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse


class ApiError(Exception):
    """Base error for application-level API responses."""

    def __init__(self, detail: str, status_code: int) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


class BadRequestError(ApiError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail=detail, status_code=400)


class NotFoundError(ApiError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail=detail, status_code=404)


class ConflictError(ApiError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail=detail, status_code=409)


class ApiKeyNotFoundError(NotFoundError):
    def __init__(self) -> None:
        super().__init__(detail="API key not found")


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error_handler(_, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
