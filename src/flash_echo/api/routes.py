from __future__ import annotations

import time

from fastapi import Body, FastAPI, HTTPException

from flash_echo.api.errors import error_payload, infer_error_param
from flash_echo.api.schemas import (
    CHAT_COMPLETION_OPENAPI_EXAMPLES,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelCard,
    ModelListResponse,
)
from flash_echo.service.filler_service import FillerReplyService


def register_routes(app: FastAPI, service: FillerReplyService) -> None:
    service_model = service.settings.service_model_name

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models", response_model=ModelListResponse)
    def list_models() -> ModelListResponse:
        created = int(time.time())
        return ModelListResponse(
            data=[
                ModelCard(
                    id=service_model,
                    created=created,
                    owned_by="flash-echo",
                )
            ]
        )

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    def chat_completions(
        request: ChatCompletionRequest = Body(openapi_examples=CHAT_COMPLETION_OPENAPI_EXAMPLES),
    ) -> ChatCompletionResponse:
        if request.stream:
            raise HTTPException(
                status_code=501,
                detail=error_payload(
                    message="Streaming is not supported yet.",
                    error_type="not_implemented_error",
                    code="stream_not_supported",
                ),
            )
        try:
            return service.create_completion(request)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=error_payload(message=str(exc), param=infer_error_param(str(exc))),
            ) from exc

