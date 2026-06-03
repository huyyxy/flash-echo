from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PersonaRoute:
    persona_tag: str
    model_version: str
    bundle_dir: Path
    enabled: bool = True


class PersonaModelRouter:
    def __init__(self, routes: dict[str, PersonaRoute], *, version: str) -> None:
        self.version = version
        self._routes = routes

    @classmethod
    def from_config(
        cls,
        payload: dict[str, Any],
        *,
        deploy_root: Path,
        template_context: dict[str, str] | None = None,
    ) -> PersonaModelRouter:
        version = str(payload.get("version", "model-router-v1"))
        raw_routes = payload.get("routes", {})
        if not isinstance(raw_routes, dict):
            raise ValueError("model router config must contain a routes object")

        routes: dict[str, PersonaRoute] = {}
        for persona_tag, route_cfg in raw_routes.items():
            if not isinstance(route_cfg, dict):
                continue
            model_version = str(route_cfg["model_version"])
            if template_context:
                model_version = model_version.format(**template_context)
            bundle_dir = deploy_root / model_version
            routes[persona_tag] = PersonaRoute(
                persona_tag=persona_tag,
                model_version=model_version,
                bundle_dir=bundle_dir,
                enabled=bool(route_cfg.get("enabled", True)),
            )
        return cls(routes, version=version)

    @classmethod
    def from_file(cls, path: Path, *, deploy_root: Path) -> PersonaModelRouter:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_config(payload, deploy_root=deploy_root)

    def resolve(self, persona_tag: str) -> PersonaRoute | None:
        route = self._routes.get(persona_tag)
        if route is None or not route.enabled:
            return None
        if not route.bundle_dir.exists():
            return None
        if not (route.bundle_dir / "model.onnx").exists():
            return None
        if not (route.bundle_dir / "inference_config.json").exists():
            return None
        return route

    def list_personas(self) -> list[str]:
        return sorted(self._routes.keys())
