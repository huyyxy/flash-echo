from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from flash_echo.api.schemas import ErrorDetail


def error_payload(
    *,
    message: str,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> dict:
    return ErrorDetail(
        status_code=400,
        message=message,
        error_type=error_type,
        param=param,
        code=code,
    ).to_payload()


def infer_error_param(message: str) -> str | None:
    if "metadata.persona_tag" in message:
        return "metadata.persona_tag"
    if "messages" in message:
        return "messages"
    return None


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def openai_http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict) and "error" in exc.detail:
            payload = exc.detail
        else:
            payload = error_payload(message=str(exc.detail))
        return JSONResponse(status_code=exc.status_code, content=payload)

