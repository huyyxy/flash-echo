from __future__ import annotations

import json
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from flash_echo.api.access_log import install_access_log
from flash_echo.api.errors import install_exception_handlers
from flash_echo.api.routes import register_routes
from flash_echo.core.cache import LruTtlCache
from flash_echo.fallback.registry import DEFAULT_FALLBACK_PREFIXES, FallbackPrefixRegistry
from flash_echo.inference.minimind3_onnx import MiniMind3ModelPool
from flash_echo.inference.persona_router import PersonaModelRouter
from flash_echo.service.filler_service import FillerReplyService, ServiceSettings
from flash_echo.static_rules.gate import DEFAULT_STATIC_RULES, StaticReplyGate


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
        bundle_dir = deploy_root / model_version
        if (bundle_dir / "model.onnx").exists() and (
            bundle_dir / "inference_config.json"
        ).exists():
            routes[persona] = {"model_version": model_version, "enabled": True}
    return {"version": "model-router-v1", "routes": routes}


def create_app() -> FastAPI:
    app = FastAPI(title="Flash Echo", version="0.1.0")
    service = build_service()
    install_access_log(app)
    install_exception_handlers(app)
    register_routes(app, service)
    return app


def main() -> None:
    host = os.getenv("FILLER_HOST", "0.0.0.0")
    port = int(os.getenv("FILLER_PORT", "8000"))
    uvicorn.run(
        "flash_echo.app:create_app",
        factory=True,
        host=host,
        port=port,
        access_log=False,
    )


if __name__ == "__main__":
    main()
