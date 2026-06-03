from __future__ import annotations

import logging
import time
from http import HTTPStatus
from typing import Awaitable, Callable

from fastapi import FastAPI, Request, Response

_access_logger = logging.getLogger("uvicorn.access")


class _TimedAccessLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, "timed_access_log", False)


def install_access_log(app: FastAPI) -> None:
    if not any(isinstance(item, _TimedAccessLogFilter) for item in _access_logger.filters):
        _access_logger.addFilter(_TimedAccessLogFilter())

    @app.middleware("http")
    async def log_request_duration(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        http_version = request.scope.get("http_version", "1.1")
        _access_logger.info(
            '%s - "%s %s HTTP/%s" %s %.2fms',
            _client_addr(request),
            request.method,
            _request_path(request),
            http_version,
            _status_line(response.status_code),
            duration_ms,
            extra={"timed_access_log": True},
        )
        return response


def _client_addr(request: Request) -> str:
    if request.client is None:
        return "-"
    return f"{request.client.host}:{request.client.port}"


def _request_path(request: Request) -> str:
    path = request.url.path
    if request.url.query:
        return f"{path}?{request.url.query}"
    return path


def _status_line(status_code: int) -> str:
    try:
        phrase = HTTPStatus(status_code).phrase
    except ValueError:
        return str(status_code)
    return f"{status_code} {phrase}"

