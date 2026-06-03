from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_TEMPLATE_PATTERN = re.compile(r"\{([^{}]+)\}")


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    pipeline_dir: Path
    runtime_dir: Path

    @classmethod
    def discover(cls, start: Path | None = None) -> "ProjectPaths":
        current = (start or Path.cwd()).resolve()
        for path in (current, *current.parents):
            if (path / "pyproject.toml").is_file() and (path / "configs").is_dir():
                return cls(
                    root=path,
                    pipeline_dir=path / "configs" / "pipelines",
                    runtime_dir=path / "configs" / "runtimes",
                )
        raise FileNotFoundError("could not find project root with pyproject.toml and configs/")


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing config file: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"config must be a YAML mapping: {path}")
    return payload


def deep_get(payload: dict[str, Any], dotted_key: str) -> Any:
    value: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"unknown template variable: {dotted_key}")
        value = value[part]
    return value


def render_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return _TEMPLATE_PATTERN.sub(lambda match: str(deep_get(context, match.group(1))), value)
    if isinstance(value, list):
        return [render_value(item, context) for item in value]
    if isinstance(value, dict):
        return {key: render_value(item, context) for key, item in value.items()}
    return value


def persona_dash(persona: str) -> str:
    return persona.replace("_", "-")
