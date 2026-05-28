import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from filler_words.api.schemas import FillerRequest, FillerResponse
from filler_words.core.cache import LruTtlCache
from filler_words.inference.classifier import HeuristicClassifier
from filler_words.service.filler_service import FillerService
from filler_words.templates.registry import TemplateRegistry


def create_app() -> FastAPI:
    app = FastAPI(title="Flash Echo", version="0.1.0")
    service = build_service()

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/filler", response_model=FillerResponse)
    def decide_filler(request: FillerRequest) -> FillerResponse:
        return service.decide(request)

    return app


def build_service() -> FillerService:
    root = Path(__file__).resolve().parents[2]
    template_path = Path(os.getenv("FILLER_TEMPLATES_PATH", root / "configs" / "templates.zh-CN.v1.json"))
    return FillerService(
        classifier=HeuristicClassifier(),
        templates=TemplateRegistry.from_file(template_path),
        cache=LruTtlCache(
            max_entries=int(os.getenv("FILLER_CACHE_MAX_ENTRIES", "10000")),
            ttl_seconds=int(os.getenv("FILLER_CACHE_TTL_SECONDS", "300")),
        ),
    )


def main() -> None:
    uvicorn.run("filler_words.app:create_app", factory=True, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
