from __future__ import annotations

import json
import os
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from filler_words.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorDetail,
    ModelCard,
    ModelListResponse,
)
from filler_words.core.cache import LruTtlCache
from filler_words.fallback.registry import DEFAULT_FALLBACK_PREFIXES, FallbackPrefixRegistry
from filler_words.inference.minimind3_onnx import MiniMind3ModelPool
from filler_words.inference.persona_router import PersonaModelRouter
from filler_words.service.filler_service import FillerReplyService, ServiceSettings
from filler_words.static_rules.gate import DEFAULT_STATIC_RULES, StaticReplyGate


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json_config(path: Path, default: dict) -> dict:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    return default


def build_service() -> FillerReplyService:
    root = project_root()
    config_dir = Path(os.getenv("FILLER_CONFIG_DIR", root / "configs"))
    deploy_root = Path(os.getenv("FILLER_DEPLOY_ROOT", root / "models" / "deploy"))

    static_gate = StaticReplyGate.from_config(
        load_json_config(config_dir / "static_rules.v1.json", DEFAULT_STATIC_RULES)
    )
    fallback_registry = FallbackPrefixRegistry.from_config(
        load_json_config(config_dir / "fallback_prefixes.v1.json", DEFAULT_FALLBACK_PREFIXES)
    )
    persona_router = PersonaModelRouter.from_config(
        load_json_config(
            config_dir / "model_router.v1.json",
            _default_model_router(deploy_root),
        ),
        deploy_root=deploy_root,
    )

    return FillerReplyService(
        static_gate=static_gate,
        persona_router=persona_router,
        fallback_registry=fallback_registry,
        model_pool=MiniMind3ModelPool(),
        cache=LruTtlCache(
            max_entries=int(os.getenv("FILLER_CACHE_MAX_ENTRIES", "10000")),
            ttl_seconds=int(os.getenv("FILLER_CACHE_TTL_SECONDS", "300")),
        ),
        settings=ServiceSettings(
            service_model_name=os.getenv("FILLER_SERVICE_MODEL", "filler-reply-minimind3"),
            max_query_chars=int(os.getenv("FILLER_MAX_QUERY_CHARS", "512")),
        ),
    )


def _default_model_router(deploy_root: Path) -> dict:
    personas = (
        "female_receptionist",
        "male_white_collar",
        "grandpa",
        "young_girl",
    )
    routes = {}
    for persona in personas:
        model_version = f"minimind3-filler-{persona.replace('_', '-')}-v1.0.0"
        if (deploy_root / model_version / "model.onnx").exists():
            routes[persona] = {"model_version": model_version, "enabled": True}
    return {"version": "model-router-v1", "routes": routes}


def create_app() -> FastAPI:
    app = FastAPI(title="Flash Echo", version="0.1.0")
    service = build_service()
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
    def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
        if request.stream:
            raise HTTPException(
                status_code=501,
                detail=_error_payload(
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
                detail=_error_payload(message=str(exc), param=_infer_error_param(str(exc))),
            ) from exc

    @app.exception_handler(HTTPException)
    async def openai_http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict) and "error" in exc.detail:
            payload = exc.detail
        else:
            payload = _error_payload(message=str(exc.detail))
        return JSONResponse(status_code=exc.status_code, content=payload)

    return app


def _error_payload(
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


def _infer_error_param(message: str) -> str | None:
    if "metadata.persona_tag" in message:
        return "metadata.persona_tag"
    if "messages" in message:
        return "messages"
    return None


def main() -> None:
    host = os.getenv("FILLER_HOST", "0.0.0.0")
    port = int(os.getenv("FILLER_PORT", "8000"))
    uvicorn.run("filler_words.app:create_app", factory=True, host=host, port=port)


if __name__ == "__main__":
    main()
