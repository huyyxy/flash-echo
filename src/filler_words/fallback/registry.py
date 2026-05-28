from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


class FallbackPrefixRegistry:
    def __init__(self, prefixes_by_locale: dict[str, dict[str, list[str]]], *, version: str) -> None:
        self.version = version
        self._prefixes_by_locale = prefixes_by_locale

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> FallbackPrefixRegistry:
        version = str(payload.get("version", "fallback-prefix-v1"))
        locale_map: dict[str, dict[str, list[str]]] = {}
        for key, value in payload.items():
            if key == "version" or not isinstance(value, dict):
                continue
            locale_map[key] = {
                persona: list(candidates)
                for persona, candidates in value.items()
                if isinstance(candidates, list) and candidates
            }
        return cls(locale_map, version=version)

    @classmethod
    def from_file(cls, path: Path) -> FallbackPrefixRegistry:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_config(payload)

    def choose(self, *, locale: str, persona_tag: str) -> str:
        locale_cfg = self._prefixes_by_locale.get(locale) or self._prefixes_by_locale.get("zh-CN", {})
        candidates = locale_cfg.get(persona_tag) or locale_cfg.get("default") or ["我想一下，"]
        return random.choice(candidates)


DEFAULT_FALLBACK_PREFIXES: dict[str, Any] = {
    "version": "fallback-prefix-v1",
    "zh-CN": {
        "female_receptionist": ["我想一下，", "我来帮你梳理一下，"],
        "male_white_collar": ["我想一下，", "这个问题可以这样看，"],
        "grandpa": ["让我想想，", "这个问题啊，"],
        "young_girl": ["嗯，我想想，", "让我理一理，"],
        "default": ["我想一下，", "我来帮你梳理一下，"],
    },
}
